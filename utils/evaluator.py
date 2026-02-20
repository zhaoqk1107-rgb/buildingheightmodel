# import torch
# import numpy as np
# from typing import List, Dict
# from tqdm import tqdm
#
#
# class Evaluator:
#     def __init__(self, device):
#         self.device = device
#
#         # 存储所有图片的预测和GT，用于全局计算 mAP
#         # 列表结构: [ {'masks': array, 'scores': array}, ... ]
#         self.all_preds = []
#         self.all_gts = []
#
#         # 像素级和高度指标累加器 (这些可以增量计算)
#         self.pixel_tp = 0
#         self.pixel_fp = 0
#         self.pixel_fn = 0
#
#         self.height_abs_diff = 0
#         self.height_sq_diff = 0
#         self.height_count = 0
#         self.height_correct_delta3 = 0
#         self.height_correct_3m = 0
#
#     @torch.no_grad()
#     def update(self, pred_instances_list: List[Dict], gt_instances_list: List[Dict],
#                pred_map: torch.Tensor, gt_map: torch.Tensor,
#                pred_height: torch.Tensor, gt_height: torch.Tensor):
#
#         # --- 1. Pixel & Height Metrics (增量计算保持不变) ---
#         pred_binary = (pred_map > 0).long()
#         gt_binary = (gt_map > 0).long()
#
#         self.pixel_tp += torch.sum((pred_binary == 1) & (gt_binary == 1)).item()
#         self.pixel_fp += torch.sum((pred_binary == 1) & (gt_binary == 0)).item()
#         self.pixel_fn += torch.sum((pred_binary == 0) & (gt_binary == 1)).item()
#
#         # Height Metrics (只在 GT 有值的地方算)
#         # mask = (gt_binary > 0)
#         # if mask.sum() > 0:
#         p_h = pred_height.clamp(min=1e-6)
#         g_h = gt_height.clamp(min=1e-6)
#
#         diff = torch.abs(p_h - g_h)
#         self.height_abs_diff += diff.sum().item()
#         self.height_sq_diff += (diff ** 2).sum().item()
#         self.height_count += (p_h>=0).sum().item()
#         self.height_correct_3m += (diff < 3.0).sum().item()
#         self.height_correct_delta3 += (torch.max(p_h / g_h, g_h / p_h) < (1.25 ** 3)).sum().item()
#
#         # --- 2. Store for Global mAP (核心修改) ---
#         # 我们只存储必要的数据到 CPU，防止爆显存
#         for pred, gt in zip(pred_instances_list, gt_instances_list):
#             # Preds
#             # 注意：我们在 inference 里已经去掉了 threshold，所以这里会有很多低分 mask
#             # 为了内存考虑，可以稍微过滤极低分的 (e.g., < 0.01)，但不要过滤 0.1
#             p_scores = pred['scores'].detach().cpu().numpy()
#             p_masks = pred['masks'].detach().cpu().numpy()  # bool numpy array
#
#             # GTs
#             g_masks = gt['masks'].detach().cpu().numpy().astype(bool)
#
#             self.all_preds.append({'masks': p_masks, 'scores': p_scores})
#             self.all_gts.append({'masks': g_masks})
#
#     def compute(self):
#         # 1. Pixel Metrics
#         eps = 1e-6
#         pixel_prec = self.pixel_tp / (self.pixel_tp + self.pixel_fp + eps)
#         pixel_rec = self.pixel_tp / (self.pixel_tp + self.pixel_fn + eps)
#         pixel_f1 = 2 * pixel_prec * pixel_rec / (pixel_prec + pixel_rec + eps)
#         pixel_iou = self.pixel_tp / (self.pixel_tp + self.pixel_fp + self.pixel_fn + eps)
#
#         # 2. Height Metrics
#         if self.height_count > 0:
#             mae = self.height_abs_diff / self.height_count
#             rmse = np.sqrt(self.height_sq_diff / self.height_count)
#             r_acc = self.height_correct_delta3 / self.height_count
#             e_acc = self.height_correct_3m / self.height_count
#         else:
#             mae, rmse, r_acc, e_acc = -1, -1, -1, -1
#
#         # 3. Global mAP Calculation
#         map_metrics = self._compute_global_map()
#
#         metrics = {
#             'Pre': pixel_prec, 'Rec': pixel_rec, 'F1': pixel_f1, 'IoU': pixel_iou,
#             'MAE': mae, 'RMSE': rmse, 'R_ACC': r_acc, 'E_ACC': e_acc
#         }
#         metrics.update(map_metrics)
#         return metrics
#
#     def _compute_global_map(self):
#         """
#         计算全局 mAP (COCO Style)
#         """
#         iou_thresholds = np.linspace(0.5, 0.95, 10)
#         ap_results = []
#
#         # 预先计算所有图片的 IoU 矩阵，加速后续循环
#         # 结构: list of [num_pred, num_gt] matrices
#         iou_matrices = []
#         for i in range(len(self.all_preds)):
#             p_masks = self.all_preds[i]['masks']  # [N, H, W]
#             g_masks = self.all_gts[i]['masks']  # [M, H, W]
#
#             if len(p_masks) == 0 or len(g_masks) == 0:
#                 iou_matrices.append(None)
#                 continue
#
#             # Flatten & Compute IoU
#             # 使用 uint8 进行矩阵乘法计算 intersection 以节省内存 (bool -> uint8)
#             p_flat = p_masks.reshape(p_masks.shape[0], -1).astype(np.uint8)
#             g_flat = g_masks.reshape(g_masks.shape[0], -1).astype(np.uint8)
#
#             intersection = np.dot(p_flat, g_flat.T)
#             p_area = p_flat.sum(1)[:, None]
#             g_area = g_flat.sum(1)[None, :]
#             union = p_area + g_area - intersection
#
#             iou_mat = intersection / (union + 1e-6)
#             iou_matrices.append(iou_mat)
#
#         # 对每个阈值计算 AP
#         for t in iou_thresholds:
#             ap = self._evaluate_single_threshold(t, iou_matrices)
#             ap_results.append(ap)
#
#         return {
#             "mAP": np.mean(ap_results),
#             "AP50": ap_results[0],
#             "AP75": ap_results[5]
#         }
#
#     def _evaluate_single_threshold(self, threshold, iou_matrices):
#         """
#         针对单个 IoU 阈值计算 Global AP
#         """
#         # 收集所有预测结果: (score, is_tp)
#         all_scores = []
#         all_tp = []
#         total_gt_count = 0
#
#         for i in range(len(self.all_preds)):
#             preds = self.all_preds[i]
#             gts = self.all_gts[i]
#             iou_mat = iou_matrices[i]  # [N, M]
#
#             num_gt = len(gts['masks'])
#             total_gt_count += num_gt
#
#             if len(preds['masks']) == 0:
#                 continue
#
#             p_scores = preds['scores']
#             num_pred = len(p_scores)
#
#             if num_gt == 0:
#                 # 所有预测都是 FP
#                 all_scores.extend(p_scores)
#                 all_tp.extend([0] * num_pred)
#                 continue
#
#             # 贪心匹配 (Greedy Matching)
#             # 1. 按分数降序排列当前图的预测
#             sorted_indices = np.argsort(-p_scores)
#             p_scores_sorted = p_scores[sorted_indices]
#             iou_mat_sorted = iou_mat[sorted_indices]  # [N, M]
#
#             gt_covered = np.zeros(num_gt, dtype=bool)
#
#             for j in range(num_pred):
#                 all_scores.append(p_scores_sorted[j])
#
#                 # 找到该预测匹配的 IoU 最大的 GT
#                 max_iou = np.max(iou_mat_sorted[j])
#                 max_idx = np.argmax(iou_mat_sorted[j])
#
#                 if max_iou >= threshold and not gt_covered[max_idx]:
#                     all_tp.append(1)
#                     gt_covered[max_idx] = True
#                 else:
#                     all_tp.append(0)
#
#         if total_gt_count == 0:
#             return 0.0
#
#         # 全局排序计算 AP
#         all_scores = np.array(all_scores)
#         all_tp = np.array(all_tp)
#
#         if len(all_scores) == 0:
#             return 0.0
#
#         # 按全局分数降序排列
#         indices = np.argsort(-all_scores)
#         all_tp = all_tp[indices]
#
#         # 计算累积 TP 和 FP
#         tp_cumsum = np.cumsum(all_tp)
#         fp_cumsum = np.cumsum(1 - all_tp)
#
#         recalls = tp_cumsum / total_gt_count
#         precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
#
#         # 11-point interpolation or area under curve (VOC07 metric)
#         # 这里使用 area under curve (COCO style)
#         mrec = np.concatenate(([0.], recalls, [1.]))
#         mpre = np.concatenate(([0.], precisions, [0.]))
#
#         # Envelope
#         for k in range(mpre.size - 1, 0, -1):
#             mpre[k - 1] = np.maximum(mpre[k - 1], mpre[k])
#
#         # Integrate
#         i = np.where(mrec[1:] != mrec[:-1])[0]
#         ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
#         return ap

import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval



class Evaluator:
    def __init__(self, device):
        self.device = device
        self.predictions = []
        self.gt_annotations = []
        self.coco_images = []  # 新增：用于存储包含宽高信息的图片字典
        self.img_id_map = {}
        self.cat_id = 1  # Building category ID
        self.image_id_counter = 0
        self.ann_id_counter = 0

        # Pixel metrics accumulators
        self.pixel_tp = 0
        self.pixel_fp = 0
        self.pixel_fn = 0

        # Height metrics accumulators
        self.height_diff_sum = 0
        self.height_sq_diff_sum = 0
        self.height_cnt = 0
        self.height_correct_3m = 0
        self.height_correct_delta3 = 0

    def update(self, pred_instances_list, gt_instances_list, pred_map, gt_map, pred_height, gt_height):
        """
        Args:
            pred_instances_list: List of dicts per image.
            gt_instances_list: List of dicts per image.
            pred_map: (B, H, W)
            gt_map: (B, H, W)
            pred_height: (B, H, W)
            gt_height: (B, H, W)
        """
        # --- 1. Pixel & Height Metrics ---
        # 确保输入是 Tensor 且在 CPU (为了累加计算不占显存，或者保持在 GPU 计算完再 item())
        # 这里为了计算效率，先在 GPU 算 sum，最后 item() 取回 CPU
        pred_binary = (pred_map > 0).long()
        gt_binary = (gt_map > 0).long()

        self.pixel_tp += torch.sum((pred_binary == 1) & (gt_binary == 1)).item()
        self.pixel_fp += torch.sum((pred_binary == 1) & (gt_binary == 0)).item()
        self.pixel_fn += torch.sum((pred_binary == 0) & (gt_binary == 1)).item()

        # Height evaluation
        # mask = (gt_binary == 1)
        # if mask.sum() > 0:
        diff = (pred_height - gt_height).abs()
        self.height_diff_sum += diff.sum().item()
        self.height_sq_diff_sum += (diff ** 2).sum().item()
        self.height_cnt += (gt_height>=0).sum().item()
        self.height_correct_3m += (diff < 3.0).sum().item()
        g_h_safe = gt_height.clamp(min=1e-6)
        p_h_safe = pred_height.clamp(min=1e-6)
        self.height_correct_delta3 += (torch.max(p_h_safe / g_h_safe, g_h_safe / p_h_safe) < (1.25 ** 3)).sum().item()

        # --- 2. Instance Metrics (Prepare for COCO format) ---
        for i, (pred_inst, gt_inst) in enumerate(zip(pred_instances_list, gt_instances_list)):
            image_id = self.image_id_counter

            # --- 关键修复：记录图片宽高 ---
            # gt_map 是 (B, H, W)，所以 gt_map[i] 是 (H, W)
            h, w = gt_map[i].shape[-2], gt_map[i].shape[-1]
            self.coco_images.append({
                "id": image_id,
                "height": int(h),
                "width": int(w)
            })

            self.image_id_counter += 1

            # --- Format Ground Truth ---
            if 'masks' in gt_inst and len(gt_inst['masks']) > 0:
                gt_masks = gt_inst['masks']
                if isinstance(gt_masks, torch.Tensor): gt_masks = gt_masks.cpu().numpy()

                for j in range(gt_masks.shape[0]):
                    mask = gt_masks[j].astype(np.uint8)
                    if mask.sum() == 0: continue  # 跳过空 mask

                    # Encode RLE
                    from pycocotools import mask as mask_utils
                    rle = mask_utils.encode(np.asfortranarray(mask))
                    rle['counts'] = rle['counts'].decode('utf-8')
                    area = float(mask_utils.area(rle))
                    bbox = list(mask_utils.toBbox(rle))

                    ann = {
                        "id": self.ann_id_counter,
                        "image_id": image_id,
                        "category_id": self.cat_id,
                        "segmentation": rle,
                        "area": area,
                        "bbox": bbox,
                        "iscrowd": 0
                    }
                    self.gt_annotations.append(ann)
                    self.ann_id_counter += 1

            # --- Format Predictions ---
            if 'masks' in pred_inst and len(pred_inst['masks']) > 0:
                p_masks = pred_inst['masks']
                p_scores = pred_inst['scores']

                if isinstance(p_masks, torch.Tensor): p_masks = p_masks.cpu().numpy()
                if isinstance(p_scores, torch.Tensor): p_scores = p_scores.cpu().numpy()

                for j in range(p_masks.shape[0]):
                    # Filter low scores
                    if p_scores[j] < 0.01: continue

                    mask = (p_masks[j] > 0.5).astype(np.uint8)
                    if mask.sum() == 0: continue

                    from pycocotools import mask as mask_utils
                    rle = mask_utils.encode(np.asfortranarray(mask))
                    rle['counts'] = rle['counts'].decode('utf-8')

                    pred = {
                        "image_id": image_id,
                        "category_id": self.cat_id,
                        "segmentation": rle,
                        "score": float(p_scores[j])
                    }
                    self.predictions.append(pred)

    def summarize(self):
        # Pixel Metrics
        eps = 1e-6
        pixel_prec = self.pixel_tp / (self.pixel_tp + self.pixel_fp + eps)
        pixel_rec = self.pixel_tp / (self.pixel_tp + self.pixel_fn + eps)
        pixel_f1 = 2 * pixel_prec * pixel_rec / (pixel_prec + pixel_rec + eps)
        pixel_iou = self.pixel_tp / (self.pixel_tp + self.pixel_fp + self.pixel_fn + eps)

        # Height Metrics
        mae = self.height_diff_sum / (self.height_cnt + eps)
        rmse = np.sqrt(self.height_sq_diff_sum / (self.height_cnt + eps))
        r_acc = self.height_correct_delta3 / self.height_cnt
        e_acc = self.height_correct_3m / self.height_cnt


        # print(f"\n📊 [Pixel-Level] IoU: {pixel_iou:.4f} | F1: {pixel_f1:.4f}")
        # print(f"📊 [Height] MAE: {mae:.4f} | RMSE: {rmse:.4f}")

        # Instance Metrics (COCO)
        if len(self.predictions) == 0 or len(self.gt_annotations) == 0:
            print("⚠️ No predictions or GT to evaluate for Instance Seg.")
            # 返回基础指标，防止外部调用报错
            return {
                "F1": pixel_f1, "IoU": pixel_iou,
                "Rec": pixel_rec, "Pre": pixel_prec,
                "MAE": mae, "RMSE": rmse,
                "mAP": 0.0, "AP50": 0.0, "AP75": 0.0,
                "rac": 0,  "eac": 0
            }

        # Build COCO GT object
        categories = [{"id": self.cat_id, "name": "building"}]

        # --- 关键修复：使用包含宽高的 self.coco_images ---
        coco_gt_dict = {
            "images": self.coco_images,
            "annotations": self.gt_annotations,
            "categories": categories
        }

        # Suppress pycocotools prints if needed, but keeping for debug is fine
        coco_gt = COCO()
        # Redirect stdout to suppress "loading annotations into memory..." if you want
        coco_gt.dataset = coco_gt_dict
        coco_gt.createIndex()

        coco_dt = coco_gt.loadRes(self.predictions)

        coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return {
            "F1": pixel_f1,
            "IoU": pixel_iou,
            "Rec": pixel_rec,
            "Pre": pixel_prec,
            "mAP": coco_eval.stats[0],  # AP @ IoU=0.50:0.95
            "AP50": coco_eval.stats[1],
            "AP75": coco_eval.stats[2],
            "MAE": mae,
            "RMSE": rmse,
            "rac": r_acc,
            "eac": e_acc
        }