import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Optional
from .gcn import GCN

# 全局缓存字典，用于存储不同尺寸的坐标网格，彻底消除重复分配和 CPU-GPU 同步阻塞
_GRID_CACHE = {}


def calculate_instance_centers(binary_masks):
    """
    计算二值掩码中每个实例的中心坐标 (已进行显存 I/O 和通信优化)。

    Args:
        binary_masks (torch.Tensor): 二值掩码张量，形状为 [batch_size, num_instance, height, width]。

    Returns:
        torch.Tensor: 形状为 [batch_size, num_instance, 2]。
    """
    bsz, num_instance, height, width = binary_masks.shape
    device = binary_masks.device

    cache_key = (height, width, device)

    # 如果该尺寸的网格尚未在当前 GPU 显存中创建，则创建并缓存
    # 直接在 GPU 上初始化，绝不经过 CPU，避免 PCIe 总线通信阻塞
    if cache_key not in _GRID_CACHE:
        # 注意：meshgrid 的参数顺序，通常 y 对应 height，x 对应 width
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device),
            indexing='ij'
        )
        # 扩展为 [1, 1, H, W] 备用，以便后续利用广播机制，而不是消耗显存去 expand
        x_grid = x_grid.unsqueeze(0).unsqueeze(0)
        y_grid = y_grid.unsqueeze(0).unsqueeze(0)
        _GRID_CACHE[cache_key] = (x_grid, y_grid)

    x_grid, y_grid = _GRID_CACHE[cache_key]

    # 计算分母，限制最小值为 1e-5 防止除零
    mask_sum = torch.sum(binary_masks, dim=[-2, -1]).clamp(min=1e-5)

    # 利用广播机制 (Broadcasting) 直接计算，[B, N, H, W] * [1, 1, H, W]
    # 省去了高昂的显存分配和扩维操作
    instance_centers_x = torch.sum(x_grid * binary_masks, dim=[-2, -1]) / mask_sum
    instance_centers_y = torch.sum(y_grid * binary_masks, dim=[-2, -1]) / mask_sum

    # 组合成 [batch_size, nq, 2] 的张量，并相对于图像高度进行归一化
    mask_centers = torch.stack([instance_centers_x, instance_centers_y], dim=-1) / height

    return mask_centers


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MaskFeatureEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=6, dropout=1):
        super().__init__()

        self.downsample = MLP(in_dim, hidden_dim * 2, hidden_dim, 2)

        self.attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))

    def forward(self, masks):
        b, n, h, w = masks.shape

        x = F.adaptive_avg_pool2d(masks, output_size=(64, 64))

        x = x.view(b, n, x.size(-2) * x.size(-1))
        x = self.downsample(x)

        x = self.norm(x)
        x, _ = self.attn(x, x, x)
        x += self.dropout(x)

        x = x.squeeze(1).view(b, n, -1)

        x = self.ffn(x)

        return x


class SelfAttentionAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, nq=200, num_heads=8, dropout=0.1):
        super(SelfAttentionAggregator, self).__init__()
        self.num_heads = num_heads
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.multihead_attention = nn.MultiheadAttention(nq, num_heads)

        self.dropout = nn.Dropout(dropout)

    def forward(self, multi_feats):
        multi_feats = self.linear(multi_feats)
        multi_feats = self.norm(multi_feats)

        multi_feats = multi_feats.permute(2, 0, 1)

        attn_output, attn_weight = self.multihead_attention(multi_feats, multi_feats, multi_feats)
        attn_output += self.dropout(attn_output)

        return attn_output.permute(1, 2, 0)


class FeatureAggregator(nn.Module):
    def __init__(self, hidden_dim=516, out_dim=256, nq=200):
        super().__init__()
        self.aggregator = SelfAttentionAggregator(hidden_dim, out_dim, nq=nq)
        self.spatial_rel_attn = GCN(nfeat=out_dim, nhid=out_dim * 2)
        self.ffn = nn.Sequential(nn.Linear(out_dim, out_dim), nn.LeakyReLU(), nn.Linear(out_dim, out_dim))
        self.norm_res = nn.LayerNorm(out_dim)

    def forward(self, pred_height, pred_logits, pred_masks, mask_embed):
        mask_centers = calculate_instance_centers(pred_masks)
        multi_feats = torch.concat([pred_height, pred_logits, mask_centers, mask_embed], dim=-1)

        height_feat_intra = self.aggregator(multi_feats)

        height_feat_gcn = self.spatial_rel_attn(height_feat_intra)
        height_feat = self.norm_res(height_feat_intra + height_feat_gcn)
        height_feat_res = self.ffn(height_feat)

        return height_feat_res