# %% Functions
# -*- encoding: utf-8 -*-
'''
@Time    :   2025/06/30 17:01:14
@Author  :   Qikang Zhao 
@Contact :   YC27963@umac.mo
@Description:   
'''
import os
import glob
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dataset import RSDataset, collate_fn
import torch.optim as optim
from model.SAM3DINOV3mask2former import SAM3DINOV3Mask2Former
from utils.matcher import HungarianMatcher
from utils.criterion import SetCriterion
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, Subset
from utils.solver import maybe_add_gradient_clipping
from utils.evaluator import Evaluator
import itertools
import yaml
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from addict import Dict
import typing
from typing import Any, List, Set
import copy
from torch.cuda.amp import autocast, GradScaler

class Trainer():
    def __init__(self, CONFIG):
        self.CONFIG = CONFIG
        self.device = CONFIG.TRAIN.device

        # 1. 创建输出目录
        os.makedirs(os.path.join(CONFIG.TRAIN.log_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(CONFIG.TRAIN.log_dir, "visualizations"), exist_ok=True)

        # 2. 数据集
        # A. 实例化“训练模式”基底 (开启增强)
        train_base_ds = RSDataset(
            CONFIG.DATASETS.datasets,
            num_sample=CONFIG.DATASETS.num_sample,
            is_train=True  # <--- 开启增强
        )
        # B. 实例化“验证模式”基底 (关闭增强)
        val_base_ds = RSDataset(
            CONFIG.DATASETS.datasets,
            num_sample=CONFIG.DATASETS.num_sample,
            is_train=False  # <--- 关闭增强，仅归一化
        )
        # C. 计算划分长度
        self.total_len = len(train_base_ds)
        self.val_len = int(self.total_len * CONFIG.TRAIN.val_split)
        self.train_len = self.total_len - self.val_len
        # D. 生成切分索引 (使用固定的 Generator 保证每次切分结果一致)
        # 我们对 train_base_ds 进行切分，主要为了获取那两组随机的 indices
        generator = torch.Generator().manual_seed(42)
        train_subset_temp, val_subset_temp = random_split(
            train_base_ds, [self.train_len, self.val_len], generator=generator
        )
        # E. 组装最终数据集
        # 训练集：使用 train_subset_temp 的索引，指向 train_base_ds (带增强)
        train_dataset = train_subset_temp
        # 验证集：使用 val_subset_temp 的索引，但强制指向 val_base_ds (无增强)
        # 这样就实现了：同一个文件的索引，在训练时读出来是增强过的，在验证时读出来是原图
        val_dataset = Subset(val_base_ds, val_subset_temp.indices)
        self.val_dataset_ref = val_dataset
        # DataLoader 保持不变
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG.TRAIN.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG.TRAIN.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=True
        )

        print(f"🚀 Start Training | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

        # 3. 模型初始化
        self.model = SAM3DINOV3Mask2Former(cfg=CONFIG).to(self.device)
        if CONFIG.TRAIN.ngpus > 1:
            self.model = nn.DataParallel(self.model)

        # # === 核心排查 1: 检查权重 ===
        # check_sam2_weights(self.model, self.device)
        # # === 核心排查 2: 强制初始化 ===
        # force_init_weights(self.model)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"🔥 Trainable Parameters: {trainable_params / 1e6:.2f} M")

        matcher = HungarianMatcher(
            cost_class=self.CONFIG.MODEL.MASK_FORMER.CLASS_WEIGHT,
            cost_mask=self.CONFIG.MODEL.MASK_FORMER.MASK_WEIGHT,
            cost_dice=self.CONFIG.MODEL.MASK_FORMER.DICE_WEIGHT,
            num_points=self.CONFIG.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        self.loss_weight_dict = {
            "loss_ce": self.CONFIG.MODEL.MASK_FORMER.CLASS_WEIGHT,
            "loss_mask": self.CONFIG.MODEL.MASK_FORMER.MASK_WEIGHT,
            "loss_dice": self.CONFIG.MODEL.MASK_FORMER.DICE_WEIGHT,
            "loss_height_l1": self.CONFIG.MODEL.MASK_FORMER.HEIGHT_WEIGHT,
            "loss_height_l2": self.CONFIG.MODEL.MASK_FORMER.HEIGHT_WEIGHT,
        }

        if self.CONFIG.MODEL.MASK_FORMER.DEEP_SUPERVISION:
            aux_weight_dict = {}
            for i in range(CONFIG.MODEL.MASK_FORMER.DEC_LAYERS):
                aux_weight_dict.update({k + f"_{i}": v for k, v in self.loss_weight_dict.items()})
            self.loss_weight_dict.update(aux_weight_dict)

        self.criterion = SetCriterion(
            num_classes = 1,
            matcher=matcher,
            weight_dict=self.loss_weight_dict,
            eos_coef=self.CONFIG.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT,
            losses= ["labels", "masks", "height"],
            num_points=self.CONFIG.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=self.CONFIG.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=self.CONFIG.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            device = self.device,
        )

        self.optimizer = self.build_optimizer()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = self.CONFIG.TRAIN.epochs, eta_min = 1e-7)
        self.train_loss_history = []
        self.val_loss_history = []

        self.scaler = GradScaler()


    # def update_loss_weights(self, epoch):
    #     """
    #     动态调整损失权重策略 (Loss Warm-up)
    #     """
    #     warmup_epochs = 5
    #     target_h_weight = self.CONFIG.MODEL.MASK_FORMER.HEIGHT_WEIGHT
    #     if epoch <= warmup_epochs:
    #         current_h_weight = target_h_weight * (epoch / warmup_epochs) + 1
    #     else:
    #         current_h_weight = target_h_weight
    #     # 更新主 Loss
    #     self.loss_weight_dict["loss_height_l1"] = current_h_weight
    #     self.loss_weight_dict["loss_height_l2"] = current_h_weight
    #     # 更新辅助 Loss (Deep Supervision)
    #     if self.CONFIG.MODEL.MASK_FORMER.DEEP_SUPERVISION:
    #         for i in range(self.CONFIG.MODEL.MASK_FORMER.DEC_LAYERS):
    #             self.loss_weight_dict[f"loss_height_l1_{i}"] = current_h_weight
    #             self.loss_weight_dict[f"loss_height_l2_{i}"] = current_h_weight
    #
    #     return current_h_weight

    def train_epoch(self, epoch):
        self.model.train()
        self.criterion.train()
        total_loss_avg = 0
        # self.update_loss_weights(epoch)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.CONFIG.TRAIN.epochs}")
        for i, (images, targets) in enumerate(pbar): # collate_fn 返回 (images, targets)
            self.optimizer.zero_grad()
            images = images.to(self.device)
            # targets 已经是 list of dicts，不需要 prepare_targets 了
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            with autocast(dtype=torch.bfloat16):
                outputs = self.model(images)
                losses = self.criterion(outputs, targets)
                total_loss = sum(losses[k] * self.loss_weight_dict[k] for k in losses.keys() if k in self.loss_weight_dict)
                total_loss = total_loss

            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.CONFIG.SOLVER.CLIP_GRADIENTS.CLIP_VALUE)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # total_loss.backward()
            total_loss_avg += total_loss.item()
            # self.optimizer.step()

            # 获取最后一层（主输出）的 loss * 权重
            w_ce = losses.get('loss_ce', torch.tensor(0.0)).item() * self.loss_weight_dict["loss_ce"]
            w_dice = losses.get('loss_dice', torch.tensor(0.0)).item() * self.loss_weight_dict["loss_dice"]
            w_mask = losses.get('loss_mask', torch.tensor(0.0)).item() * self.loss_weight_dict["loss_mask"]
            w_l1 = losses.get('loss_height_l1', torch.tensor(0.0)).item() * self.loss_weight_dict["loss_height_l1"]
            w_l2 = losses.get('loss_height_l2', torch.tensor(0.0)).item() * self.loss_weight_dict["loss_height_l2"]

            pbar.set_postfix({
                "📉Loss": f"{total_loss.item():.2f}",
                "Class": f"{w_ce:.2f}",
                "Mask": f"{(w_mask + w_dice):.2f}",
                "Height": f"{(w_l1 + w_l2):.2f}",
            })
            torch.cuda.empty_cache()

        avg_loss = total_loss_avg / len(self.train_loader)
        return avg_loss

    def plot_loss_curve(self):
        """
        绘制 Train 和 Val 的 Loss 曲线
        """
        try:
            plt.figure(figsize=(10, 6))
            epochs = range(1, len(self.train_loss_history) + 1)

            # 绘制 Training Loss
            plt.plot(epochs, self.train_loss_history, marker='.', linestyle='-', color='b', label='Train Loss')

            # 绘制 Validation Loss (如果有)
            if len(self.val_loss_history) > 0:
                # 对齐 Epoch (Val 可能不是每个 Epoch 都有，这里简化假设每个 Epoch 都有)
                # 如果 Val 是间隔做的，需要处理 x 轴
                val_epochs = [i * self.CONFIG.TRAIN.eval_interval for i in range(1, len(self.val_loss_history) + 1)]
                plt.plot(val_epochs, self.val_loss_history, marker='o', linestyle='--', color='r', label='Val Loss')

            plt.title('Training & Validation Loss')
            plt.xticks(epochs)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)

            save_path = os.path.join(self.CONFIG.TRAIN.log_dir, "loss_curve.png")
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            print(f"Error plotting loss curve: {e}")

    def train_model(self):
        best_score = 0.6
        for epoch in range(self.CONFIG.TRAIN.epochs):
            loss = self.train_epoch(epoch)
            # === 新增：记录并绘制 Loss 曲线 ===
            self.train_loss_history.append(loss)
            self.scheduler.step()
            if (epoch + 1) % self.CONFIG.TRAIN.eval_interval == 0:
                metrics, val_loss = self.evaluate()
                score = metrics['F1']
                self.val_loss_history.append(val_loss)
                # self.scheduler.step(score)
                self.validate_and_visualize(epoch + 1)

                torch.save(self.model.state_dict(), os.path.join(self.CONFIG.TRAIN.log_dir, "checkpoints", f"epoch_{epoch}.pth"))

                if score > best_score:
                    best_score = score
                    torch.save(self.model.state_dict(), os.path.join(self.CONFIG.TRAIN.log_dir, "checkpoints", "best.pth"))
                    print(f"⭐ New Best F1: {best_score:.2f}")

            self.plot_loss_curve()

    def evaluate(self):
        self.model.eval()
        # 1. 实例化评估器 (每轮评估新建一个，因为新 Evaluator 没有 reset 方法)
        evaluator = Evaluator(self.device)
        val_loss_total = 0.0

        # 遍历验证集
        for i, (images, targets) in enumerate(tqdm(self.val_loader, desc="Validating")):
            if i > 100: break

            images = images.to(self.device)
            # 将 targets 移动到 GPU 用于 Loss 计算
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            with torch.no_grad():
                outputs = self.model(images)
            losses = self.criterion(outputs, targets)
            batch_loss = sum(losses[k] * self.loss_weight_dict[k] for k in losses.keys() if k in self.loss_weight_dict)
            val_loss_total += batch_loss.item()

            h, w = images.shape[-2:]

            # 3. 准备 Batch 数据传给 Evaluator
            # 新 Evaluator.update 接受列表形式的 batch 数据
            batch_pred_inst = []  # List[Dict]
            batch_gt_inst = []  # List[Dict]
            batch_pred_map = []  # List[Tensor] -> Stack later
            batch_gt_map = []  # List[Tensor] -> Stack later
            batch_pred_height = []  # List[Tensor] -> Stack later
            batch_gt_height = []  # List[Tensor] -> Stack later

            for k in range(len(targets)):
                # --- GT 处理 ---
                gt_mask = targets[k]['masks'].to(self.device)  # (N, H, W)
                gt_height_map = targets[k]['height_map'].to(self.device).squeeze()  # (H, W)

                # 构造 GT Instance Dict (符合新 Evaluator 要求)
                batch_gt_inst.append({
                    "masks": gt_mask > 0.5,  # Boolean Tensor
                    "labels": targets[k]['labels'].to(self.device)
                })

                # 构造 GT Semantic Map (从实例 mask 合并)
                gt_sem_map = torch.zeros((h, w), device=self.device)
                for j, m in enumerate(gt_mask):
                    gt_sem_map[m > 0.5] = 1  # 语义分割只关心前景(1)和背景(0)
                batch_gt_map.append(gt_sem_map)
                batch_gt_height.append(gt_height_map)

                # 对单张图片进行后处理推理
                instance_list, instance_map, height_map = self.instance_inference(
                    outputs['pred_logits'][k],
                    outputs['pred_masks'][k],
                    outputs['pred_heights'][k],
                    target_size=(h, w),
                )

                batch_pred_inst.append(instance_list)
                pred_sem_map = (instance_map > 0).float()
                batch_pred_map.append(pred_sem_map)
                batch_pred_height.append(height_map)

            # 4. 调用 Update (Batch 级)
            evaluator.update(
                pred_instances_list=batch_pred_inst,
                gt_instances_list=batch_gt_inst,
                pred_map=torch.stack(batch_pred_map),  # (B, H, W)
                gt_map=torch.stack(batch_gt_map),  # (B, H, W)
                pred_height=torch.stack(batch_pred_height),  # (B, H, W)
                gt_height=torch.stack(batch_gt_height)  # (B, H, W)
            )

        # 5. 总结指标 (使用 summarize 替代 compute)
        metrics = evaluator.summarize()
        avg_val_loss = val_loss_total / (i + 1)

        # 打印结果 (根据新 Evaluator 返回的 Key 进行调整)
        if metrics is not None:
            print(f"📊 F1: {metrics.get('F1', 0):.3f}")
            print(f"📊 IoU: {metrics.get('IoU', 0):.3f}")
            print(f"📊 Recall: {metrics.get('Rec', 0):.3f}")
            print(f"📊 Precsion: {metrics.get('Pre', 0):.3f}")
            print(f"📊 mAP: {metrics.get('mAP', 0):.3f}")
            print(f"📊 AP50: {metrics.get('AP50', 0):.3f}")
            print(f"📊 AP75: {metrics.get('AP75', 0):.3f}")
            print(f"📊 MAE: {metrics.get('MAE', 0):.3f}")
            print(f"📊 RMSE: {metrics.get('RMSE', 0):.3f}")
            print(f"📊 RAC: {metrics.get('rac', 0):.3f}")
            print(f"📊 EAC: {metrics.get('eac', 0):.3f}")
        else:
            print("⚠️ Metrics calculation returned None (possibly no predictions).")
            metrics = {'F1': 0.0}  # 防止报错

        return metrics, avg_val_loss

    @torch.no_grad()
    def instance_inference(self, logits, mask_pred, height_pred, target_size, threshold=0.01):
        """
        Refined Instance Inference (Standard Top-K Strategy):
        1. 选取 Top-K (100) 个 Query。
        2. 对于 Evaluation (instance_list): 保留所有 Top-K 结果（仅使用极低阈值过滤噪声），交给 Evaluator 计算 Recall。
        3. 对于 Visualization (instance_map): 使用像素级竞争 (Argmax) 处理重叠，threshold 仅作为背景判定标准。
        4. 结合了 Instance 的 Top-K 筛选和 Panoptic 的 Overlap Filtering 机制。
        彻底解决 "Argmax Artifacts" (甜甜圈/同心圆) 问题。
        """

        # 1. 上采样 & 基础数据准备
        mask_pred = F.interpolate(
            mask_pred.unsqueeze(0),
            size=target_size,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # 2. 计算分数 (Class Score * Mask Quality)
        # 不管分数多低，先取前 100 名。依靠 mAP 评估去惩罚低分误检。
        class_prob = logits.sigmoid().squeeze(-1)
        num_queries = class_prob.shape[0]
        topk_num = min(100, num_queries)
        scores_per_image, topk_indices = class_prob.topk(topk_num, sorted=False)
        # 根据 Top-K 提取数据
        mask_pred = mask_pred[topk_indices]
        height_pred = height_pred[topk_indices]
        if height_pred.dim() > 1: height_pred = height_pred.squeeze(1)

        # 3. 计算 Mask Quality Score (Official Strategy)
        mask_pred_sigmoid = mask_pred.sigmoid()
        mask_pred_binary = (mask_pred > 0).float() # 官方阈值 0.0 (logits)
        # 最终分数 = 分类概率 * Mask质量(二值化Mask内的平均置信度)
        mask_scores_per_image = (mask_pred_sigmoid.flatten(1) * mask_pred_binary.flatten(1)).sum(1) / (mask_pred_binary.flatten(1).sum(1) + 1e-6)
        final_scores = scores_per_image * mask_scores_per_image

        # 4A. 输出 Instance List
        # 保留所有 Top-K 给 evaluator 去算 recall
        keep = final_scores > 0
        instance_list = {
            "masks": mask_pred_binary[keep].bool(), # [M, H, W]
            "scores": final_scores[keep], # [M]
            "labels": torch.zeros_like(final_scores[keep], dtype=torch.long),
            "heights": height_pred[keep]
        }

        # 4B. 输出 Instance Map (用于可视化 & 生成最终产品, 解决"同心圆"问题)
        h, w = target_size
        final_instance_map = torch.zeros((h, w), dtype=torch.int32, device=self.device)
        final_height_map = torch.zeros((h, w), dtype=torch.float32, device=self.device)
        if final_scores.shape[0] == 0:
            return instance_list, final_instance_map, final_height_map

        # # --- 像素竞争逻辑 (Vectorized) ---只有通过了 threshold 的 Query 才有资格参加像素竞争
        candidate_mask = final_scores > threshold
        comp_scores = final_scores[candidate_mask] # [K]
        comp_masks_probs = mask_pred_sigmoid[candidate_mask] # [K, H, W]
        comp_heights = height_pred[candidate_mask]  # [K]
        # 构造概率张量 [M, H, W]
        prob_map = comp_scores.view(-1, 1, 1) * comp_masks_probs
        # 构造背景层参与竞争：如果某像素上所有 Mask 的分数都低于 threshold，则该像素判为背景
        bg_prob = torch.full((1, h, w), threshold, device=self.device)
        # 拼接: [Background, Instance_1, Instance_2, ...] # 0是背景, 1是第一个candidate. # 维度变为 [K+1, H, W]
        all_probs = torch.cat([bg_prob, prob_map], dim=0)
        # Argmax: 0 是背景，1~K 是实例
        winner_indices = all_probs.argmax(dim=0)

        # 面积比率过滤 (Panoptic Logic from bdhnet)
        # 目的：如果一个 Mask 在竞争中丢失了大部分面积（变成了甜甜圈的外圈），则将其剔除
        # 我们遍历所有 Top-K 实例
        num_candidates = comp_scores.shape[0]
        for k in range(1, num_candidates + 1):
            # 竞争胜出面积 (Won Area)
            won_mask = (winner_indices == k) & (comp_masks_probs[k - 1] >= 0.5)
            won_area = won_mask.sum().item()
            # 过滤逻辑：
            # 1. 必须赢得了至少 1 个像素
            # 2. 必须原本就有像素
            # 3. 赢得的比例必须足够高 (防止甜甜圈外圈)
            # overlap_threshold，默认值 0.6, 意味着如果一个 mask 赢下的面积不到它原始面积的 60%，它就会被丢弃
            original_area = (comp_masks_probs[k - 1] >= 0.5).sum().item() # 原始预测面积 (Original Area),对应 mask_pred_binary[k-1]
            overlap_threshold = 0.5
            if won_area > 0 and original_area > 0:
                if won_area < overlap_threshold * original_area:
                    continue  # 剔除！这个 Mask 只是个“外圈”，丢弃它，这部分像素变回背景
                final_instance_map[won_mask] = k  # 这里 k 是 1~K 的唯一ID
                final_height_map[won_mask] = comp_heights[k - 1]
        return instance_list, final_instance_map, final_height_map


    def validate_and_visualize(self, epoch):
        try:
            self.model.eval()
            rand_idx = random.randint(0, self.val_len - 1)
            item = self.val_dataset_ref[rand_idx]
            batch_data = [item]
            images, targets = collate_fn(batch_data)

            images = images.to(self.device)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                outputs = self.model(images)
                _, instance_map, height_map = self.instance_inference(
                    outputs['pred_logits'][0],
                    outputs['pred_masks'][0],
                    outputs['pred_heights'][0],
                    target_size=images.shape[-2:],
                )


            # --- 数据准备 ---
            img_vis = images[0].cpu().numpy().transpose(1, 2, 0)
            img_vis = (img_vis * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)

            gt_h_vis = targets[0]['height_map'].squeeze().cpu().numpy()
            pred_h_vis = height_map.cpu().numpy()

            # --- 绘图 (1行 5列) ---
            plt.figure(figsize=(14, 5))

            # 1. Google Image
            plt.subplot(1, 3, 1)
            plt.imshow(img_vis)
            plt.title(f"Google Image")
            plt.colorbar()
            plt.axis('off')

            # 2. GT Height
            plt.subplot(1, 3, 2)
            plt.imshow(gt_h_vis, cmap='RdYlBu_r')
            plt.title("Reference Height")
            plt.colorbar()
            plt.axis('off')

            # 5. Pred Height (预测实例高程)
            plt.subplot(1, 3, 3)
            plt.imshow(pred_h_vis, cmap='RdYlBu_r')
            plt.title("Predicted Height")
            plt.colorbar()
            plt.axis('off')

            plt.savefig(os.path.join(self.CONFIG.TRAIN.log_dir, "visualizations", f"epoch_{epoch}.png"))
            plt.close()

        except Exception as e:
            print(f"可视化失败: {e}")
            import traceback
            traceback.print_exc()

    def build_optimizer(self):
        weight_decay_norm = self.CONFIG.SOLVER.get('WEIGHT_DECAY_NORM', 0.0)
        weight_decay_embed = self.CONFIG.SOLVER.get('WEIGHT_DECAY_EMBED', 0.0)

        defaults = {}
        defaults["lr"] = self.CONFIG.SOLVER.BASE_LR
        defaults["weight_decay"] = self.CONFIG.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[typing.Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in self.model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "sam2_encoder" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * self.CONFIG.SOLVER.BACKBONE_MULTIPLIER
                if "relative_position_bias_table" in module_param_name or "absolute_pos_embed" in module_param_name:
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = self.CONFIG.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    self.CONFIG.SOLVER.CLIP_GRADIENTS.ENABLED
                    and self.CONFIG.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = self.CONFIG.SOLVER.OPTIMIZER
        current_lr = self.CONFIG.SOLVER.BASE_LR

        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, current_lr, momentum=0.9, weight_decay=0.0001)
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, current_lr)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")

        if not self.CONFIG.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(self.CONFIG, optimizer)

        return optimizer



if __name__ == "__main__":
    with open('utils/config-SAM3DINOV3mask2former.yaml', 'r', encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)
    trainer = Trainer(Dict(CONFIG))
    trainer.train_model()


# %%
