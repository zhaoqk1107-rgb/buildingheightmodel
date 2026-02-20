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
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum("nc,mc->nm", neg, (1 - targets)
                                                                  )

    return loss / hw


def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
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


# 修改 utils/matcher.py

def batch_sigmoid_tversky_loss(inputs, targets, alpha=0.7, beta=0.3):
    """
    Tversky Loss for Matcher (Pairwise Calculation)
    计算所有 Prediction 和所有 GT 之间的两两 Tversky Cost。

    Args:
        inputs: [num_queries, num_points] (Logits)
        targets: [num_gt, num_points] (0/1)
    Returns:
        cost_matrix: [num_queries, num_gt]
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)  # [N, P]
    targets = targets.flatten(1)  # [M, P]

    # 1. 计算 Intersection (TP) -> [N, M]
    # 使用 einsum 进行矩阵乘法，计算每一对 (n, m) 的重叠部分
    tp = torch.einsum("nc,mc->nm", inputs, targets)

    # 2. 计算各自的总和
    p_sum = inputs.sum(dim=1)  # [N] Predicted Area
    t_sum = targets.sum(dim=1)  # [M] Ground Truth Area

    # 3. 推导 FP 和 FN (利用广播机制)
    # FP (误检) = 预测总面积 - 重叠面积
    # [N, 1] - [N, M] -> [N, M]
    fp = p_sum[:, None] - tp

    # FN (漏检) = GT总面积 - 重叠面积
    # [1, M] - [N, M] -> [N, M]
    fn = t_sum[None, :] - tp

    # 4. 计算 Tversky 系数
    tversky = tp / (tp + alpha * fp + beta * fn + 1e-6)

    # 5. 返回 Cost (1 - Tversky)
    return 1 - tversky


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"
        self.num_points = num_points  # 这里的 points 通常比训练少，BDHNet默认好像是 12544

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
            # [关键修改] 生成随机采样点 (Uniform Random)
            # Matcher 阶段通常只需要随机采样，不需要不确定性采样，因为此时还不知道谁匹配谁
            # num_points = self.num_points
            # step = int(torch.sqrt(torch.tensor(num_points)).item())
            # # 生成归一化网格坐标 [0, 1]
            # xv, yv = torch.meshgrid(torch.linspace(0, 1, step), torch.linspace(0, 1, step), indexing='ij')
            # grid_coords = torch.stack([yv, xv], dim=-1).reshape(1, -1, 2).to(out_mask.device)  # [1, step*step, 2]
            # # 如果点数不够，补齐；多了截断（通常 num_points 设置为 step平方 即可，例如 12544=112*112）
            # if grid_coords.shape[1] < num_points:
            #     padding = torch.rand(1, num_points - grid_coords.shape[1], 2, device=out_mask.device)
            #     point_coords = torch.cat([grid_coords, padding], dim=1)
            # else:
            #     point_coords = grid_coords[:, :num_points, :]
            # # 添加轻微抖动，避免死板
            # point_coords = point_coords + (torch.rand_like(point_coords) - 0.5) * (1.0 / step)
            # point_coords = point_coords.clamp(0, 1)

            # 在预测 Mask 上采样
            # out_mask: [num_queries, 1, H, W] -> sample -> [num_queries, 1, num_points] -> squeeze -> [num_queries, num_points]
            out_mask_sampled = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False
            ).squeeze(1)

            # 在 GT Mask 上采样
            # tgt_mask: [num_gt, 1, H, W] -> sample -> [num_gt, num_points]
            tgt_mask_sampled = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask_sampled = out_mask_sampled.float()
                tgt_mask_sampled = tgt_mask_sampled.float()

                # 计算 Cost (基于采样点)
                # cost_mask = batch_sigmoid_ce_loss(out_mask_sampled, tgt_mask_sampled)
                # Mask2Former/BDHNet 这里通常用 Focal Loss
                cost_mask = batch_sigmoid_ce_loss(out_mask_sampled, tgt_mask_sampled)
                # cost_dice = batch_dice_loss(out_mask_sampled, tgt_mask_sampled)
                cost_dice = batch_sigmoid_tversky_loss(out_mask_sampled, tgt_mask_sampled, alpha=0.7, beta=0.3)

            C = (
                    self.cost_mask * cost_mask
                    + self.cost_class * cost_class
                    + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @torch.no_grad()
    def forward(self, outputs, targets):
        return self.memory_efficient_forward(outputs, targets)