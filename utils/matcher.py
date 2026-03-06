# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast


def point_sample(input, point_coords, **kwargs):
    """
    同 criterion.py 中的实现
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets))

    return loss / hw


# def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
#     hw = inputs.shape[1]
#
#     prob = inputs.sigmoid()
#     focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
#         inputs, torch.ones_like(inputs), reduction="none"
#     )
#     focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
#         inputs, torch.zeros_like(inputs), reduction="none"
#     )
#     if alpha >= 0:
#         focal_pos = focal_pos * alpha
#         focal_neg = focal_neg * (1 - alpha)
#
#     loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum("nc,mc->nm", focal_neg, (1 - targets))
#
#     return loss / hw




class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.cost_center = 2
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"
        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []

        for b in range(bs):
            # 1. Class Cost 保持不变 (Softmax)
            out_prob = outputs["pred_logits"][b].softmax(-1)
            tgt_ids = targets[b]["labels"]
            cost_class = -out_prob[:, tgt_ids]

            # 2. 🚨 修复：改回 PointRend 点匹配 🚨
            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            tgt_mask = targets[b]["masks"].to(out_mask)  # [num_gt, H_gt, W_gt] 原尺寸高清 Mask

            # 核心：生成均匀的纯随机点进行匹配（匹配阶段无需计算 uncertainty，纯随机最快）
            # 提取 12544 个随机点坐标
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)

            # [num_gt, num_points] 从高清 GT 提取 12544 个像素的值
            tgt_mask = point_sample(
                tgt_mask.unsqueeze(1).float(),
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False
            ).squeeze(1)

            # [num_queries, num_points] 从预测特征图提取这 12544 个位置的值
            out_mask = point_sample(
                out_mask.unsqueeze(1),
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()

                # 在这 12544 个点上计算代价，避免了任何降采样带来的边缘膨胀
                cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                cost_dice = batch_dice_loss(out_mask, tgt_mask)

            # 合并 Cost 进行匈牙利匹配
            C = (self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice)
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @torch.no_grad()
    def forward(self, outputs, targets):
        return self.memory_efficient_forward(outputs, targets)