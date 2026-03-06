import torch
import torch.nn.functional as F
from torch import nn
from .misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list, get_world_size


# ==================================================================================
# PointRend 核心工具函数 (复现自 detectron2)
# ==================================================================================

def point_sample(input, point_coords, **kwargs):
    """
    从 input feature map 中根据 point_coords 进行采样。
    Args:
        input: [N, C, H, W]
        point_coords: [N, P, 2], 范围在 [0, 1] 之间
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)

    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)

    if add_dim:
        output = output.squeeze(3)
    return output


def calculate_uncertainty(logits):
    """
    计算二分类 logits 的不确定性。
    logits 越接近 0 (概率 0.5)，不确定性越大。
    我们用 -abs(logits) 来表示，值越大表示越不确定。
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def get_uncertain_point_coords_with_randomness(
        logits, num_points, oversample_ratio, importance_sample_ratio
):
    """
    PointRend 的核心采样策略：
    1. 随机采样更多的点 (num_points * oversample_ratio)。
    2. 计算这些点的不确定性。
    3. 选择最不确定的那些点 (based on importance_sample_ratio)。
    4. 补充一些纯随机点，保持分布的多样性。
    """
    assert logits.shape[1] == 1, "Only support binary mask for point sampling"
    batch_size = logits.shape[0]
    num_sampled = int(num_points * importance_sample_ratio)
    point_coords = torch.rand(batch_size, num_points, 2, device=logits.device)

    if num_sampled <= 0:
        return point_coords

    # 1. 随机采样更多候选点
    num_uncertain_points = int(num_points * oversample_ratio)
    candidate_coords = torch.rand(batch_size, num_uncertain_points, 2, device=logits.device)

    # 2. 获取这些点的预测值
    candidate_logits = point_sample(logits, candidate_coords, align_corners=False)

    # 3. 计算不确定性 & 选择 Top-K
    uncertainty_map = calculate_uncertainty(candidate_logits)
    _, idx = torch.topk(uncertainty_map.squeeze(1), num_sampled, dim=1)

    # 4. 提取坐标
    idx = idx.unsqueeze(-1).repeat(1, 1, 2)
    chosen_coords = torch.gather(candidate_coords, 1, idx)

    # 5. 拼接：最重要的点 + 剩余的纯随机点
    if num_sampled < num_points:
        random_coords = torch.rand(batch_size, num_points - num_sampled, 2, device=logits.device)
        point_coords = torch.cat([chosen_coords, random_coords], dim=1)
    else:
        point_coords = chosen_coords

    return point_coords


def dice_loss(inputs, targets, num_masks):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_ce_loss(inputs, targets, num_masks):
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
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    原版 Mask2Former 必备的分类 Focal Loss
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks



class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, device):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.device = device
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        empty_weight = torch.ones(self.num_classes + 1, device=self.device)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_masks):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()  # [B, num_queries, num_cls+1]
        idx = self._get_src_permutation_idx(indices) # shape [n]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        return {"loss_ce": loss_ce}

    def loss_masks(self, outputs, targets, indices, num_masks):
        """
        [关键修改] Mask Loss 计算：不再使用全图插值，而是使用 PointRend 采样策略。
        这样可以让模型专注于优化“边缘”等难点，从而让输出更锐利（直角化）。
        """
        assert "pred_masks" in outputs

        # 1. 提取匹配好的 Pred Mask
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"][src_idx]  # [N_matched, H_pred, W_pred]
        src_masks = src_masks.unsqueeze(1)  # [N_matched, 1, H_pred, W_pred]

        # 2. 提取匹配好的 GT Mask
        target_masks = [t["masks"] for t in targets]
        target_masks, _ = nested_tensor_from_tensor_list(target_masks).decompose()
        target_masks = target_masks.to(src_masks)[tgt_idx]  # [N_matched, H_gt, W_gt]
        target_masks = target_masks.unsqueeze(1)  # [N_matched, 1, H_gt, W_gt]

        # 3. [PointRend]
        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks.detach(),
                self.num_points,  # 12544
                self.oversample_ratio,  # 3.0
                self.importance_sample_ratio  # 0.75
            )

            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        # 6. 计算 Point-wise Loss
        losses = {
            "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }
        del point_logits, target_masks, src_masks
        return losses

    # def loss_masks(self, outputs, targets, indices, num_masks):
    #     """
    #     [修改说明] 移除 PointRend 采样，使用 Dense (密集) 方式计算 Mask Loss。
    #     保证遥感图像中每一个微小的建筑物像素都能参与梯度回传。
    #     """
    #     assert "pred_masks" in outputs
    #
    #     # 1. 提取匹配好的 Pred Mask
    #     src_idx = self._get_src_permutation_idx(indices)
    #     tgt_idx = self._get_tgt_permutation_idx(indices)
    #
    #     # [N_matched, H_pred, W_pred]
    #     src_masks = outputs["pred_masks"][src_idx]
    #
    #     # 2. 提取匹配好的 GT Mask
    #     target_masks = [t["masks"] for t in targets]
    #     target_masks, _ = nested_tensor_from_tensor_list(target_masks).decompose()
    #     # [N_matched, H_gt, W_gt]
    #     target_masks = target_masks.to(src_masks)[tgt_idx]
    #
    #     # 3. 尺寸对齐：将高分辨率的 GT Mask 降采样到 Pred Mask 的尺寸 (通常是 1/4)
    #     # 降采样 GT 比上采样 Pred 更节省显存，且不影响小目标检测
    #     target_masks = F.interpolate(
    #         target_masks.unsqueeze(1).float(),
    #         size=src_masks.shape[-2:],
    #         mode="bilinear"
    #     ).squeeze(1)
    #
    #     # 4. 展平张量准备计算 Loss
    #     # [N_matched, H_pred * W_pred]
    #     src_masks = src_masks.flatten(1)
    #     target_masks = target_masks.flatten(1)
    #
    #     # 5. 计算 Dense Point-wise Loss
    #     losses = {
    #         "loss_mask": sigmoid_ce_loss(src_masks, target_masks, num_masks),
    #         "loss_dice": dice_loss(src_masks, target_masks, num_masks),
    #     }
    #
    #     del target_masks, src_masks
    #     return losses

    def loss_height(self, outputs, targets, indices, num_masks):
        assert "pred_heights" in outputs
        src_heights = outputs["pred_heights"]
        idx = self._get_src_permutation_idx(indices)
        matched_pred_heights = src_heights[idx]
        target_heights = torch.cat([t["height_instances"][J].squeeze(1) for t, (_, J) in zip(targets, indices)]).to(self.device)
        if matched_pred_heights.numel() == 0:
            return {"loss_height_l1": src_heights.sum() * 0.0, "loss_height_l2": src_heights.sum() * 0.0}
        del src_heights
        matched_pred_heights = torch.log1p(matched_pred_heights.clamp(min=0))
        target_heights   = torch.log1p(target_heights.clamp(min=0))
        return {"loss_height_l1": F.l1_loss(matched_pred_heights, target_heights, reduction='sum')/num_masks,
                "loss_height_l2": F.smooth_l1_loss(matched_pred_heights, target_heights, beta=1.0, reduction='sum') / num_masks}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'height': self.loss_height
            # 'boundary': self.loss_boundary,  # 注册新 Loss
        }
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)


        # ====== 插入这段调试代码 ======
        if self.training:
            src_idx = indices[0][0]  # 取出第一个 Batch 的 Query 分配索引
            unique_queries = torch.unique(src_idx)
            # print(f"\n[Debug] 当前图像有 {len(targets[0]['labels'])} 个真实建筑物。")
            # print(f"[Debug] 共有 {len(unique_queries)} 个不同的 Query 参与了匹配。")
            if len(unique_queries) < len(targets[0]['labels']) * 0.5:
                print("🚨 警告：大量真实建筑物被分配给了同一个 Query，Query 已崩塌！")

        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor([num_masks], dtype=torch.float, device=self.device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses