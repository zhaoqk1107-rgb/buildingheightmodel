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


    @staticmethod
    def _mask_centers(mask: torch.Tensor, threshold: float = 0.5):
        """
        计算 mask 的归一化中心坐标。
        输入: [N, H, W]，输出: [N, 2] (x, y), 范围 [0, 1].
        对空 mask 使用图像中心作为回退值，避免 NaN。
        """
        n, h, w = mask.shape
        bin_mask = (mask > threshold).float()
        mass = bin_mask.sum(dim=(1, 2), keepdim=False)

        ys = torch.linspace(0, 1, steps=h, device=mask.device, dtype=mask.dtype).view(1, h, 1)
        xs = torch.linspace(0, 1, steps=w, device=mask.device, dtype=mask.dtype).view(1, 1, w)

        cy = (bin_mask * ys).sum(dim=(1, 2)) / (mass + 1e-6)
        cx = (bin_mask * xs).sum(dim=(1, 2)) / (mass + 1e-6)

        centers = torch.stack([cx, cy], dim=-1)
        fallback = torch.full_like(centers, 0.5)
        valid = (mass > 0).unsqueeze(-1)
        centers = torch.where(valid, centers, fallback)
        return centers


    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        indices = []

        for b in range(bs):
            # 1. Class Cost
            out_prob = outputs["pred_logits"][b].softmax(-1)
            tgt_ids = targets[b]["labels"]
            cost_class = -out_prob[:, tgt_ids]

            # 2. Mask Cost (使用 Point Sampling)
            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            tgt_mask = targets[b]["masks"].to(out_mask)  # [num_gt, H_gt, W_gt]


            # 统一形状：[N, 1, H, W]
            out_mask = out_mask.unsqueeze(1)
            tgt_mask = tgt_mask.unsqueeze(1)

            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False
            ).squeeze(1)

            # 在 GT Mask 上采样
            # tgt_mask: [num_gt, 1, H, W] -> sample -> [num_gt, num_points]
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()

                # 计算 Cost (基于采样点)
                cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                cost_dice = batch_dice_loss(out_mask, tgt_mask)

                pred_centers = self._mask_centers(outputs["pred_masks"][b].sigmoid())
                tgt_centers = self._mask_centers(targets[b]["masks"].to(out_mask).float())
                cost_center = torch.cdist(pred_centers, tgt_centers, p=1)


            C = (self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice + self.cost_center * cost_center)

            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @torch.no_grad()
    def forward(self, outputs, targets):
        return self.memory_efficient_forward(outputs, targets)