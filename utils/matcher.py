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


def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    hw = inputs.shape[1]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum("nc,mc->nm", focal_neg, (1 - targets))

    return loss / hw




class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"
        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []

        for b in range(bs):
            # 1. Class Cost (彻底重构为 Focal Cost)
            # 此时的 outputs["pred_logits"] 已经被修复为 [B, num_queries, 1]
            out_prob = outputs["pred_logits"][b].sigmoid()  # [num_queries, 1]

            alpha = 0.25
            gamma = 2.0

            # 计算匹配到背景的代价 (负样本代价) 和匹配到目标的代价 (正样本代价)
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())

            # 代价差值：匹配该目标的收益
            cost_class = pos_cost_class - neg_cost_class  # [num_queries, 1]

            # 获取当前图片中的 GT 数量
            num_gt = targets[b]["labels"].shape[0]
            # 因为只有 1 个前景类别(类别0)，任意 Query 分配给任意 GT 的分类代价都是相同的
            # 我们将 [num_queries, 1] 的代价向量广播成 [num_queries, num_gt] 矩阵
            if num_gt > 0:
                cost_class = cost_class.repeat(1, num_gt)
            else:
                cost_class = torch.empty((num_queries, 0), device=out_prob.device)

            # 2. Mask Cost (保持原有的优秀实现)
            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            tgt_mask = targets[b]["masks"].to(out_mask)  # [num_gt, H_gt, W_gt]

            out_mask = out_mask.unsqueeze(1)
            tgt_mask = tgt_mask.unsqueeze(1)

            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)

            out_mask_sampled = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False
            ).squeeze(1)

            tgt_mask_sampled = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask_sampled = out_mask_sampled.float()
                tgt_mask_sampled = tgt_mask_sampled.float()

                # cost_mask = batch_sigmoid_focal_loss(out_mask_sampled, tgt_mask_sampled)
                cost_mask = batch_sigmoid_ce_loss(out_mask_sampled, tgt_mask_sampled)
                cost_dice = batch_dice_loss(out_mask_sampled, tgt_mask_sampled)

            C = (self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice)

            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @torch.no_grad()
    def forward(self, outputs, targets):
        return self.memory_efficient_forward(outputs, targets)