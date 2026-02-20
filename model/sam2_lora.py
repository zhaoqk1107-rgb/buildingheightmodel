import math
import torch
import torch.nn as nn
from typing import Dict


class _LoRA_Linear(nn.Module):
    """
    针对 Linear 层的 LoRA 包装器
    """

    def __init__(self, original_linear: nn.Linear, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original_linear = original_linear
        self.dim_in = original_linear.in_features
        self.dim_out = original_linear.out_features
        self.r = r
        self.scaling = alpha / r

        # LoRA 矩阵
        self.lora_A = nn.Parameter(torch.zeros(self.dim_in, r))
        self.lora_B = nn.Parameter(torch.zeros(r, self.dim_out))

        # 初始化: B=0 保证初始状态与原模型完全一致
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 原有路径 (Frozen)
        base_out = self.original_linear(x)
        # LoRA 路径 (Trainable)
        # x: [B, ..., In], A: [In, r], B: [r, Out]
        # x @ A -> [B, ..., r] @ B -> [B, ..., Out]
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out


class LoRA_SAM2(nn.Module):
    """
    专门适配 SAM 2 (Hiera) 的 LoRA 注入器
    """

    def __init__(self, sam2_model, r=16):
        super().__init__()
        self.sam2_model = sam2_model
        self.r = r

        # 1. 冻结所有参数
        for param in self.sam2_model.parameters():
            param.requires_grad = False

        # 2. 遍历 Image Encoder 注入 LoRA
        self.inject_lora(self.sam2_model.image_encoder)

        # 打印一下注入了多少层，确保逻辑生效
        trainable_params = sum(p.numel() for p in self.sam2_model.parameters() if p.requires_grad)
        print(f"✅ LoRA Injected. Trainable Params: {trainable_params / 1e6:.2f}M")

    def inject_lora(self, module):
        # 使用 named_children 遍历子模块
        # 需要转为 list，因为我们在遍历过程中修改了 module 的结构
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                # 策略：覆盖 Attention (qkv, proj) 和 MLP (fc1, fc2)
                # 增加 'fc' 和 'mlp' 关键词以覆盖更多层
                target_keywords = ["qkv", "proj", "fc", "linear", "mlp"]
                if any(k in name.lower() for k in target_keywords):
                    new_layer = _LoRA_Linear(child, r=self.r)

                    # [修复 Bug 1] 处理 nn.Sequential 的特殊情况
                    if isinstance(module, nn.Sequential):
                        module[int(name)] = new_layer
                    else:
                        setattr(module, name, new_layer)

                    # 激活 LoRA 参数梯度
                    new_layer.lora_A.requires_grad = True
                    new_layer.lora_B.requires_grad = True
            else:
                # 递归
                self.inject_lora(child)

    def forward(self, x):
        # [修复 Bug 3] 确保 features 始终被定义
        features = {}

        # 获取 SAM2 Backbone 输出
        out_dict = self.sam2_model.image_encoder(x)

        if isinstance(out_dict, dict) and "backbone_fpn" in out_dict:
            fpn_feats = out_dict["backbone_fpn"]
            # 确保 FPN 特征足够
            if len(fpn_feats) >= 3:
                # 对应 stride 4, 8, 16
                # 使用 clone() 或直接引用均可，主要确保 device 正确
                features["res2"] = fpn_feats[0]
                features["res3"] = fpn_feats[1]
                features["res4"] = fpn_feats[2]
            else:
                print("⚠️ Warning: SAM2 FPN features insufficient.")
        else:
            print("⚠️ Warning: SAM2 output format unexpected.")

        return features