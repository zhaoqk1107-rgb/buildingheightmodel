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



class LoG_Filter(nn.Module):
    def __init__(self, in_channels, kernel_size=7, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size

        # 1. 生成 LoG 核
        # grid范围: -3 到 3 (对于 k=7)
        pad = kernel_size // 2
        coords = torch.arange(-pad, pad + 1, dtype=torch.float32)
        x, y = torch.meshgrid(coords, coords, indexing='ij')

        # LoG 公式 (对应论文图中的公式)
        # LoG(x,y) = -1/(pi*sigma^4) * (1 - (x^2+y^2)/(2*sigma^2)) * exp(...)
        # 注意：通常我们需要翻转算子符号以获得正响应，或者让网络自己适应。
        # 这里使用标准 LoG 近似
        r2 = x ** 2 + y ** 2
        sigma2 = sigma ** 2
        gamma = 1 / (math.pi * sigma2 ** 2)
        kernel = gamma * (2 - r2 / sigma2) * torch.exp(-r2 / (2 * sigma2))

        # 归一化 (让核的总和为0，这是拉普拉斯算子的特性)
        kernel = kernel - kernel.mean()

        # 2. 构造成卷积权重 [Out, In/Groups, k, k]
        # 使用 Depthwise Conv，每个通道独立滤波
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(in_channels, 1, 1, 1)

        self.register_buffer('weight', kernel)
        self.groups = in_channels
        self.padding = pad

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=self.padding, groups=self.groups)


class LEA_Module(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 多尺度 LoG 分支
        self.log_0_5 = LoG_Filter(channels, kernel_size=3, sigma=0.5)
        self.log_1_0 = LoG_Filter(channels, kernel_size=5, sigma=1.0)
        self.log_2_0 = LoG_Filter(channels, kernel_size=7, sigma=2.0)

        # 融合卷积 (Concat 3xChannels -> Channels)
        self.aggregator = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True)  # 文中提到用 SiLU
        )
        # 残差融合
        # F_res = F_init + BN(SiLU(F_agg)) -> 文中公式(3)
        # 但通常 Conv 也在 BN 之前。这里 aggregator 已经包含了 conv+bn+silu
        # 我们稍微调整以匹配常规 ResBlock 逻辑
        self.final_bn = nn.BatchNorm2d(channels)  # 可选

    def forward(self, x):
        # x: [B, C, H, W]
        # 1. 多尺度滤波
        f1 = self.log_0_5(x)
        f2 = self.log_1_0(x)
        f3 = self.log_2_0(x)
        # 2. 拼接 & 聚合
        f_cat = torch.cat([f1, f2, f3], dim=1)  # [B, 3C, H, W]
        f_agg = self.aggregator(f_cat)
        # 3. 残差连接 (对应论文公式 3)
        return x + f_agg


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


class BoundaryHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 简单的卷积层，从特征图中提取边缘
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
    def forward(self, x):
        return self.conv(x)

## Reference:
## SAM3UNET; DINOV3UNET


class SAM3DINOV3Mask2Former(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.sam_encoder = SAM3Encoder(checkpoint_path=cfg.SAM.checkpoint_path, img_size=512)
        self.dino_encoder = DinoEncoder(model_name="dinounet_l", pretrained_path=cfg.DINO.checkpoint_path, features_per_stage=[256, 256, 256, 256])
        self.lea_modules = nn.ModuleDict()
        self.cbam_modules = nn.ModuleDict()
        self.fusion_modules = nn.ModuleDict()
        feature_levels = ['res2', 'res3', 'res4', 'res5']
        for i, level in enumerate(feature_levels):
            self.fusion_modules[level] = nn.Sequential(
                nn.Conv2d(128 + 256, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            self.cbam_modules[level] = CBAM(256)
            self.lea_modules[level] = LEA_Module(256)
        self.backbone_feature_shape = dict()
        for i, stride in zip([2, 3, 4, 5], [4, 8, 16, 32]):
            self.backbone_feature_shape[f'res{i}'] = Dict({'channel': 256, 'stride': stride})
        self.sem_seg_head = MaskFormerHead(self.backbone_feature_shape, cfg)
        self.adb_height_head = AdaBinsHead(cfg.TRAIN.min_height, cfg.TRAIN.max_height)
        self.boundary_head = BoundaryHead(256)

    def forward(self, x):
        sam_features = self.sam_encoder(x)
        dino_features = self.dino_encoder(x)        # DINO Features: {'res2', 'res3', 'res4', 'res5'}

        # self._visualize_comparison(x, sam_features, dino_features, save_name="vis_feature_debug.png")

        enhanced_features = {}        # 逐层 融合 & 增强
        for i, level in enumerate(['res2', 'res3', 'res4', 'res5']):
            f_sam = sam_features[i]
            f_dino = F.max_pool2d(dino_features[i], kernel_size=4, stride=4)
            f_fused = self.fusion_modules[level](torch.cat([f_sam, f_dino], dim=1))
            enhanced_features[level] = self.lea_modules[level](self.cbam_modules[level](f_fused))

        # Decoder
        outputs = self.sem_seg_head(enhanced_features)
        outputs["pred_heights"] = self.adb_height_head(outputs.pop("height_feats"), outputs.pop("out_bins"))
        outputs["pred_boundaries"] = self.boundary_head(outputs.pop("mask_feats"))

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