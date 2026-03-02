import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from scipy.optimize import linear_sum_assignment
import math
from addict import Dict
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from .base.CBAM import CBAM
from .dinounet_training import DinoEncoder
from model.SAM3UNet.SAM3UNet import SAM3Encoder
from model.base.encoders import get_encoder

class MaskFormerHead(nn.Module):
    def __init__(self, input_shape, cfg):
        super().__init__()
        self.pixel_decoder = MSDeformAttnPixelDecoder(input_shape,
                                                transformer_dropout = cfg.MODEL.MASK_FORMER.DROPOUT,
                                                transformer_nheads = cfg.MODEL.MASK_FORMER.NHEADS,
                                                transformer_dim_feedforward = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_DIM_FEEDFORWARD,
                                                transformer_enc_layers=cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS,
                                                conv_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
                                                mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
                                                transformer_in_features = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES,
                                                common_stride = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE)

        self.predictor = MultiScaleMaskedTransformerDecoder(in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM,
                                                        num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                                                        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
                                                        num_queries = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
                                                        nheads = cfg.MODEL.MASK_FORMER.NHEADS,
                                                        dim_feedforward = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
                                                        dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1,
                                                        pre_norm = cfg.MODEL.MASK_FORMER.PRE_NORM,
                                                        mask_dim = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
                                                        mask_classification=True,
                                                        enforce_input_project = False,
                                                        num_bins = cfg.TRAIN.n_bins,
                                                        min_height=cfg.TRAIN.min_height,
                                                        max_height = cfg.TRAIN.max_height)


    def forward(self, features, mask=None):
        mask_features, _, multi_scale_features = self.pixel_decoder.forward_features(features)
        predictions = self.predictor(multi_scale_features, mask_features, mask)
        predictions["mask_feats"] = mask_features
        return predictions


class AdaBinsHead(nn.Module):
    def __init__(self,min_height, max_height):
        super().__init__()
        self.max_height = max_height
        self.min_height = min_height

    def forward(self, height_feats, output_bins):
        # height_feats和output_bins: [b, 200(nq), 256]
        bin_widths = (self.max_height - self.min_height) * output_bins # [b, 200(nq), 256]
        bin_widths = F.pad(bin_widths, (1, 0), mode='constant', value=0) # [B, Q, num_bins + 1] -> 第一个是0
        bin_edges = torch.cumsum(bin_widths, dim=-1) + self.min_height  # [B, Q, num_bins + 1]
        centers = 0.5 * (bin_edges[..., :-1] + bin_edges[..., 1:])  # [b, 200(nq), 256]
        probs = F.softmax(height_feats, dim=-1)  # [b, nq, 256] #  关键修复：对 height_feats 进行 Softmax，将其转换为概率分布
        pred_heights = torch.sum(probs * centers, dim=-1) # [b, nq]
        return pred_heights


## Reference:
## SAM3UNET; DINOV3UNET
# ── 特征融合辅助模块 ────────────────────────────────────────────────────────────

class SEGate(nn.Module):
    """
    Squeeze-Excitation 通道门控。
    在三路特征拼接之前对 CNN 分支单独加权，防止 DINO 强预训练信号
    在 1×1 线性压缩中覆盖 CNN 的局部纹理/边界特征。
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gate(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * w

class FusionBlock(nn.Module):
    def __init__(self, in_ch: int, cnn_ch: int, sam_ch: int,
                 dino_ch: int, out_ch: int = 256):
        super().__init__()
        # 三路各自独立门控，互不干扰
        self.cnn_gate  = SEGate(cnn_ch)
        self.sam_gate  = SEGate(sam_ch)
        self.dino_gate = SEGate(dino_ch)

        self.compress = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1,
                      groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, cnn_feat, sam_feat, dino_feat):
        # 三路分别自适应调权，消除激活幅度不一致问题
        return self.compress(torch.cat([
            self.cnn_gate(cnn_feat),
            self.sam_gate(sam_feat),
            self.dino_gate(dino_feat),
        ], dim=1))

class SAM3DINOV3Mask2Former(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # ── 编码器 ─────────────────────────────────────────────────────────
        # ResNet-50 depth=5: get_encoder 返回 6 个张量，取 [2:] 得 4 层：
        #   [0] (B,  256, 128, 128)  → res2
        #   [1] (B,  512,  64,  64)  → res3
        #   [2] (B, 1024,  32,  32)  → res4
        #   [3] (B, 2048,  16,  16)  → res5
        self.cnn_encoder = get_encoder('resnet50', in_channels=3, depth=5, weights='imagenet')
        # SAM3: 输出 list[4]
        #   [0] (B, 128, 128, 128)  → res2
        #   [1] (B, 128,  64,  64)  → res3
        #   [2] (B, 128,  32,  32)  → res4
        #   [3] (B, 128,  16,  16)  → res5
        self.sam_encoder = SAM3Encoder(checkpoint_path=cfg.SAM.checkpoint_path, img_size=512)
        # DINO-v3 (最后1个stage解冻): 输出 list[4]
        #   [0] (B, 256, 512, 512)  ← 不使用
        #   [1] (B, 256, 256, 256)  ← 不使用
        #   [2] (B, 256, 128, 128)  → res2
        #   [3] (B, 256,  64,  64)  → res3；插值↓→res4/res5
        self.dino_encoder = DinoEncoder(model_name="dinounet_l", pretrained_path=cfg.DINO.checkpoint_path, features_per_stage=[256, 256, 256, 256])
        # ── SE 门控融合层 ──────────────────────────────────────────────────
        # 三路拼接通道数（CNN + SAM + DINO）：
        #   res2:  256 + 128 + 256 =  640，CNN 分支通道 = 256
        #   res3:  512 + 128 + 256 =  896，CNN 分支通道 = 512
        #   res4: 1024 + 128 + 256 = 1408，CNN 分支通道 = 1024
        #   res5: 2048 + 128 + 256 = 2432，CNN 分支通道 = 2048
        self.fusion_modules = nn.ModuleDict({
            'res2': FusionBlock(640, cnn_ch=256, sam_ch=128, dino_ch=256),
            'res3': FusionBlock(896, cnn_ch=512, sam_ch=128, dino_ch=256),
            'res4': FusionBlock(1408, cnn_ch=1024, sam_ch=128, dino_ch=256),
            'res5': FusionBlock(2432, cnn_ch=2048, sam_ch=128, dino_ch=256),
        })
        # ── Mask2Former Head ────────────────────────────────────────────────
        self.backbone_feature_shape = {
            f'res{i}': Dict({'channel': 256, 'stride': s})
            for i, s in zip([2, 3, 4, 5], [4, 8, 16, 32])}
        self.sem_seg_head = MaskFormerHead(self.backbone_feature_shape, cfg)
        self.adb_height_head = AdaBinsHead(cfg.TRAIN.min_height, cfg.TRAIN.max_height)

        # 从最深层特征中提取整张图像唯一的投影偏角信息
        self.global_view_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )

    @staticmethod
    def _align(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """双线性插值对齐到 ref 的空间尺寸，保留梯度，避免 max_pool 信息损失。"""
        if src.shape[-2:] == ref.shape[-2:]:
            return src
        return F.interpolate(
            src, size=ref.shape[-2:], mode='bilinear', align_corners=False
        )

    def forward(self, x):
        cnn_all  = self.cnn_encoder(x)[2:]   # list[4]
        sam_all  = self.sam_encoder(x)        # list[4]
        dino_all = self.dino_encoder(x)       # list[4]      # DINO Features: {'res2', 'res3', 'res4', 'res5'}

        # CNN: [2:] 后索引 0~3
        c2 = cnn_all[0]    # (B,  256, 128, 128)
        c3 = cnn_all[1]    # (B,  512,  64,  64)
        c4 = cnn_all[2]    # (B, 1024,  32,  32)
        c5 = cnn_all[3]    # (B, 2048,  16,  16)

        # SAM: 索引 0~3，与 res2~res5 一一对应
        s2 = sam_all[0]    # (B, 128, 128, 128)
        s3 = sam_all[1]    # (B, 128,  64,  64)
        s4 = sam_all[2]    # (B, 128,  32,  32)
        s5 = sam_all[3]    # (B, 128,  16,  16)

        # DINO: idx2→res2；idx3→res3，插值对齐到 res4/res5
        d2 = dino_all[2]   # (B, 256, 128, 128)
        d3 = dino_all[3]   # (B, 256,  64,  64)

        # ── 2. 逐层对齐 + SE融合 ──────────────────────────────────────────
        # res2: 三路均已在 128×128，无需对齐
        f2 = self.fusion_modules['res2'](c2, s2, d2)
        #   输入 (B,256,128,128)+(B,128,128,128)+(B,256,128,128) → (B,256,128,128)

        # res3: 三路均已在 64×64，无需对齐
        f3 = self.fusion_modules['res3'](c3, s3, d3)
        #   输入 (B,512,64,64)+(B,128,64,64)+(B,256,64,64) → (B,256,64,64)

        # res4: 目标 32×32；DINO d3(64²) 插值↓2x
        d4 = self._align(d3, s4)             # (B, 256, 32, 32)
        f4 = self.fusion_modules['res4'](c4, s4, d4)
        #   输入 (B,1024,32,32)+(B,128,32,32)+(B,256,32,32) → (B,256,32,32)

        # res5: 目标 16×16；DINO d3(64²) 插值↓4x
        d5 = self._align(d3, s5)             # (B, 256, 16, 16)
        f5 = self.fusion_modules['res5'](c5, s5, d5)
        #   输入 (B,2048,16,16)+(B,128,16,16)+(B,256,16,16) → (B,256,16,16)

        # ── 3. 送入 Mask2Former Decoder ────────────────────────────────────
        enhanced_features = {
            'res2': f2,   # (B, 256, 128, 128)  stride=4
            'res3': f3,   # (B, 256,  64,  64)  stride=8
            'res4': f4,   # (B, 256,  32,  32)  stride=16
            'res5': f5,   # (B, 256,  16,  16)  stride=32
        }

        # self._visualize_comparison(x, sam_features, dino_features, save_name="vis_feature_debug.png")

        # 获取图像全局视角的偏移向量 (每米产生的归一化坐标偏移)
        # 用 Tanh 限制极端值，假设 100米建筑极限偏移不超过图像边长的 20%
        # 这里缩放系数 0.002 意味着：(100m * 0.002 = 0.2)，在 512 分辨率下大概是 100 像素

        # Decoder
        outputs = self.sem_seg_head(enhanced_features)
        outputs["pred_heights"] = self.adb_height_head(outputs.pop("height_feats"), outputs.pop("out_bins"))
        if "aux_outputs" in outputs:# 计算中间层的高度预测
            for aux_out in outputs["aux_outputs"]:
                aux_out["pred_heights"] = self.adb_height_head(aux_out.pop("height_feats"), aux_out.pop("out_bins")) # 这里的 height_feats 也是经过每一层的 FeatureAggregator (GCN) 计算出来的
        return outputs


    @torch.no_grad()
    def _visualize_comparison(self, x, sam_feats, dino_feats, save_name="vis_feature_debug.png"):
        """
        可视化输入图像、SAM特征和DINO特征的对比
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # 1. 准备原图 (取 Batch 中的第 0 张)
        # 假设输入经过了 ImageNet 标准化，这里进行反标准化以便人眼观察
        img = x[0].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img * std + mean).clip(0, 1)
        # 2. 准备特征图 (以 res3 / stride=8 为例，这是中层特征，包含语义和纹理)
        # SAM Features
        # sam_feats['res3'] shape: [B, C, H/8, W/8]
        sam_map = sam_feats[0][0].mean(dim=0)
        # DINO Features (dino_features 是 list)
        # dino_feats[1] 对应 res3
        dino_map = dino_feats[0][0].mean(dim=0)
        # 3. 统一上采样到原图尺寸以便观察
        h, w = x.shape[-2:]
        sam_map = F.interpolate(sam_map.view(1, 1, *sam_map.shape), size=(h, w), mode='bilinear',
                                align_corners=False).squeeze().cpu().numpy()
        dino_map = F.interpolate(dino_map.view(1, 1, *dino_map.shape), size=(h, w), mode='bilinear',
                                 align_corners=False).squeeze().cpu().numpy()

        # 4. 绘图
        plt.figure(figsize=(15, 5))

        # Input Image
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title("Input Satellite Image")
        plt.axis('off')

        # SAM Features
        plt.subplot(1, 3, 2)
        plt.imshow(sam_map, cmap='viridis')  # 使用 viridis 热力图
        plt.title("SAM2 Features")
        plt.axis('off')

        # DINO Features
        plt.subplot(1, 3, 3)
        plt.imshow(dino_map, cmap='inferno')  # 使用 inferno 热力图
        plt.title("DINOv3 Features")
        plt.axis('off')

        # 保存到本地，而不是 show，防止卡住训练流
        plt.savefig(save_name)
        plt.close()
        # print(f"🖼️ Visualization saved to {save_name}")