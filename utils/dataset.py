# # -*- encoding: utf-8 -*-
# '''
# @Time    :   2025/06/30 17:01:14
# @Author  :   Qikang Zhao
# @Contact :   YC27963@umac.mo
# @Description:
# '''
# import os
# import sys
# from glob import glob
# import numpy as np
# import rasterio
# import math
# import warnings
# warnings.filterwarnings('ignore')
# from scipy.ndimage import zoom
# import torch.nn as nn
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader, random_split
# import cv2
# # from torchvision import transforms
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
#
#
# class RSDataset(Dataset):
#     def __init__(self, dataset_names, size=512, num_sample=-1, is_train=True):
#         self.is_train = is_train
#         self.img_paths = list()
#         self.label_paths = list()
#         for ds in dataset_names:
#             print(ds)
#             self.img_paths += sorted(glob(os.path.join(f"datasets/{ds}/image/*.tif")))
#             self.label_paths += sorted(glob(os.path.join(f"datasets/{ds}/height/*.tif")))
#         assert len(self.img_paths) == len(self.label_paths), "images != heights"
#
#         if num_sample > 0 and num_sample < len(self.img_paths):
#             import random
#             random.seed(42)
#             all_indices = list(range(len(self.img_paths)))
#             random.shuffle(all_indices)
#             selected_indices = all_indices[:num_sample]
#             self.img_paths = [self.img_paths[i] for i in selected_indices]
#             self.label_paths = [self.label_paths[i] for i in selected_indices]
#             self.upscale_factor = 2
#
#         self.size = size
#         # SAM2 期望输入是 RGB，范围 0-1，然后减去以下均值除以方差
#         # self.normalize = transforms.Normalize(
#         #     mean=[0.485, 0.456, 0.406],
#         #     std=[0.229, 0.224, 0.225]
#         # )
#
#         pixel_mean = [0.485, 0.456, 0.406]
#         pixel_std = [0.229, 0.224, 0.225]
#         if self.is_train:
#             self.transform = A.Compose([
#                 # === 几何变换 (最为关键) ===
#                 # 随机翻转 (水平+垂直)
#                 A.HorizontalFlip(p=0.5),
#                 A.VerticalFlip(p=0.5),
#                 # 随机 90 度旋转 (0, 90, 180, 270)，这是处理建筑物朝向的神器
#                 A.Rotate(limit=180, p=1.0),
#                 # 随机调整亮度和对比度
#                 A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
#                 # 这里的 Normalize 会把数据除以 255 并减均值除方差
#                 A.Normalize(mean=pixel_mean, std=pixel_std),
#                 ToTensorV2()
#             ])
#         else:
#             # 验证集只需要归一化和转 Tensor，不需要乱转
#             self.transform = A.Compose([
#                 A.Normalize(mean=pixel_mean, std=pixel_std),
#                 ToTensorV2()
#             ])
#
#     def __len__(self):
#         return len(self.img_paths)
#
#     def __getitem__(self, idx):
#         # 1. Read Label (Height Map)
#         with rasterio.open(self.label_paths[idx]) as src:
#             height_data = src.read(1).astype(np.float32)
#         valid_mask = ~(np.isnan(height_data) | np.isinf(height_data) | (height_data < 0) | (height_data==255) | (height_data > 1000))
#         height_data = np.where(valid_mask, height_data, 0)
#
#         # 2. Read Image
#         with rasterio.open(self.img_paths[idx]) as src:
#             img_data = np.stack([src.read(i) for i in range(1, 4)])
#         # 归一化到 [0, 1]
#         # img_tensor = torch.from_numpy(img_data) / 255.0
#         # raw_img_tensor = img_tensor.clone()
#         # img_tensor = self.normalize(img_tensor)
#         img_data = np.transpose(img_data, (1, 2, 0))  # (H, W, 3)
#         augmented = self.transform(image=img_data, mask=height_data)
#         img_tensor = augmented['image'] # Tensor (3, 512, 512)
#         height_data_aug = augmented['mask'] # Tensor or Numpy (512, 512)
#         if isinstance(height_data_aug, torch.Tensor):
#             height_data_aug = height_data_aug.numpy()
#
#         # 3. Generate Instance Masks & Heights
#         label_binary = (height_data_aug > 0).astype(np.uint8)
#         # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#         # label_binary = cv2.erode(label_binary, kernel, iterations=1)
#         # 4. Generate Instance Masks & Heights
#         num_objs, labels_im = cv2.connectedComponents(label_binary)
#
#         masks = []
#         classes = []
#         height_instances = []
#         for i in range(1, num_objs):
#             mask_loc = (labels_im == i)
#             if mask_loc.sum() > 1:
#                 mask_i = mask_loc.astype(np.float32)
#                 masks.append(mask_i)
#                 classes.append(0)  # Class 0: Building (Evaluator里我们统一成了0)
#
#                 # Calculate mean height for this instance
#                 avg_h = height_data[mask_loc].mean()
#                 height_instances.append(avg_h)
#
#         # 处理没有建筑物的情况 (防止DataLoader报错)
#         if len(masks) == 0:
#             masks_tensor = torch.zeros((0, img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.float32)
#             classes_tensor = torch.zeros(0, dtype=torch.long)
#             height_instances_tensor = torch.zeros((0, 1), dtype=torch.float32)
#         else:
#             masks_tensor = torch.tensor(np.stack(masks), dtype=torch.float32)
#             classes_tensor = torch.tensor(classes, dtype=torch.long)
#             height_instances_tensor = torch.tensor(height_instances, dtype=torch.float32).unsqueeze(1)
#
#         output = {
#             "images": img_tensor,
#                 # (B, 3, H, W)
#                 # 标准的谷歌影像RGB张量
#             "masks": masks_tensor,
#                 # (B, N, H, W) 实例分割的核心真值。N 代表这张图片里有多少个建筑物。
#                 # 比如图里有 5 个房子，形状就是 (B, 5, 512, 512)。
#                 # 用途: HungarianMatcher会拿模型预测的200个Query Mask与这 N 个真值 Mask 进行二分图匹配（IoU/Dice）。
#                 # 匹配上之后，计算 Mask Loss。
#             "labels": classes_tensor,
#                 #   (B, N,) . 每个实例的类别标签。
#                 #  用途: 告诉模型“刚才那个 Mask 是一个建筑物”。
#             "height_instances": height_instances_tensor,
#                 # 每个建筑物实例的平均高度。(B, N, 1)
#                 # 用途: 用于计算loss_height。
#                 # 为什么要在这里算？: 在 Dataset 里用 CPU 预先算好每个实例的平均高度（avg_h = height_data[mask_loc].mean()），
#                 # 比在训练循环里用 GPU 去做 Mask 索引和求平均要快得多，也更稳定。
#             "height_map": torch.from_numpy(height_data_aug).unsqueeze(0).float()
#                 # 原始的高度图真值。(1, H, W)
#                 # 主要用于 验证 (Validation) 和 可视化。
#                 #  在训练 Loss 计算中，其实有了height_instances就够了，
#                 #  但为了画出直观的对比图（Pred vs GT），我们保留这张整图。
#             }
#         self.plot = False
#         if self.plot:
#             self._visualize_debug(idx, img_tensor, height_data_aug, label_binary, masks_tensor)
#
#         return output
#
#     def _visualize_debug(self, idx, img_tensor, height_data, label_binary, masks_tensor):
#         """调试用绘图函数"""
#         # (C, H, W) -> (H, W, C)
#         img_vis = img_tensor.permute(1, 2, 0).numpy()
#         mean = np.array([0.485, 0.456, 0.406])
#         std = np.array([0.229, 0.224, 0.225])
#         img_vis = std * img_vis + mean
#         img_vis = np.clip(img_vis, 0, 1)
#
#         # 合并所有实例 Mask 查看是否有重叠或遗漏
#         if masks_tensor.shape[0] > 0:
#             combined_mask = masks_tensor.sum(dim=0).numpy()
#         else:
#             combined_mask = np.zeros_like(height_data)
#
#         import matplotlib.pyplot as plt
#
#         plt.figure(figsize=(20, 5))
#
#         plt.subplot(141)
#         plt.imshow(img_vis)
#         plt.title(f'ID: {idx} | Input RGB')
#         plt.axis('off')
#
#         plt.subplot(142)
#         plt.imshow(height_data, cmap='jet')
#         plt.title('GT Height Map')
#         plt.axis('off')
#
#         plt.subplot(143)
#         plt.imshow(label_binary, cmap='gray')
#         plt.title('Binary Footprint')
#         plt.axis('off')
#
#         plt.subplot(144)
#         plt.imshow(combined_mask, cmap='tab20')
#         plt.title(f'Instance Masks (N={masks_tensor.shape[0]})')
#         plt.axis('off')
#
#         save_name = f'debug_dataset_{idx}.png'
#         plt.tight_layout()
#         plt.savefig(save_name, dpi=600)
#         plt.show()
#         # plt.close()
#         # print(f"🐛 Vis trainning data image saved to {save_name}")
#
#
# def collate_fn(batch):
#     images = torch.stack([item['images'] for item in batch])
#     targets = []
#     for item in batch:
#         target_dict = {
#             "masks": item['masks'],
#             "labels": item['labels'],
#             "height_instances": item['height_instances'],
#             "height_map": item['height_map']
#         }
#         targets.append(target_dict)
#
#     return images, targets


# -*- encoding: utf-8 -*-
'''
@Time    :   2025/06/30 17:01:14
@Author  :   Qikang Zhao
@Contact :   YC27963@umac.mo
@Description:   Modified RSDataset with 2x upsampling and 4-way tiling.
'''
import os
import sys
from glob import glob
import numpy as np
import rasterio
import math
import warnings

warnings.filterwarnings('ignore')
from scipy.ndimage import zoom
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RSDataset(Dataset):
    def __init__(self, dataset_names, size=512, num_sample=-1, is_train=True):
        self.is_train = is_train
        self.img_paths = list()
        self.label_paths = list()
        for ds in dataset_names:
            print(f"Loading dataset: {ds}")
            self.img_paths += sorted(glob(os.path.join(f"datasets/{ds}/image/*.tif")))
            self.label_paths += sorted(glob(os.path.join(f"datasets/{ds}/height/*.tif")))

        assert len(self.img_paths) == len(self.label_paths), "images != heights"

        if num_sample > 0 and num_sample < len(self.img_paths):
            import random
            random.seed(42)
            all_indices = list(range(len(self.img_paths)))
            random.shuffle(all_indices)
            selected_indices = all_indices[:num_sample]
            self.img_paths = [self.img_paths[i] for i in selected_indices]
            self.label_paths = [self.label_paths[i] for i in selected_indices]

        self.size = size  # 目标尺寸 512
        self.upscale_size = size * 2  # 上采样到 1024

        pixel_mean = [0.485, 0.456, 0.406]
        pixel_std = [0.229, 0.224, 0.225]

        if self.is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=180, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.Normalize(mean=pixel_mean, std=pixel_std),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=pixel_mean, std=pixel_std),
                ToTensorV2()
            ])

    def __len__(self):
        # 每个样本对应 4 个切片
        return len(self.img_paths) * 4

    def __getitem__(self, idx):
        # 1. 计算原始索引和切片位置
        original_idx = idx // 4
        tile_pos = idx % 4  # 0:左上, 1:右上, 2:左下, 3:右下

        # 2. 读取 Label (Height Map)
        with rasterio.open(self.label_paths[original_idx]) as src:
            height_raw = src.read(1).astype(np.float32)

        # 数据清洗
        valid_mask = ~(np.isnan(height_raw) | np.isinf(height_raw) | (height_raw < 0) | (height_raw == 255) | (
                    height_raw > 1000))
        height_raw = np.where(valid_mask, height_raw, 0)

        # 3. 读取 Image
        with rasterio.open(self.img_paths[original_idx]) as src:
            img_raw = np.stack([src.read(i) for i in range(1, 4)])
        img_raw = np.transpose(img_raw, (1, 2, 0))  # (H, W, 3)

        # 4. 上采样到 1024x1024
        # 图像用双线性插值，标签用最邻近插值
        img_large = cv2.resize(img_raw, (self.upscale_size, self.upscale_size), interpolation=cv2.INTER_LINEAR)
        height_large = cv2.resize(height_raw, (self.upscale_size, self.upscale_size), interpolation=cv2.INTER_NEAREST)

        # 5. 切分 512x512
        # 计算切片坐标
        row = tile_pos // 2
        col = tile_pos % 2
        y1, y2 = row * self.size, (row + 1) * self.size
        x1, x2 = col * self.size, (col + 1) * self.size

        img_tile = img_large[y1:y2, x1:x2, :]
        height_tile = height_large[y1:y2, x1:x2]

        # 6. 增强处理
        augmented = self.transform(image=img_tile, mask=height_tile)
        img_tensor = augmented['image']
        height_data_aug = augmented['mask']
        if isinstance(height_data_aug, torch.Tensor):
            height_data_aug = height_data_aug.numpy()

        # 7. 生成实例 Mask
        label_binary = (height_data_aug > 0).astype(np.uint8)
        label_eroded = cv2.erode(label_binary, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        num_objs, labels_im = cv2.connectedComponents(label_eroded)

        masks = []
        classes = []
        height_instances = []
        for i in range(1, num_objs):
            mask_loc = (labels_im == i)
            if mask_loc.sum() > 1:
                mask_i = mask_loc.astype(np.float32)
                masks.append(mask_i)
                classes.append(0)  # Class 0: Building

                # 计算增强后切片内该实例的平均高度
                avg_h = height_data_aug[mask_loc].mean()
                height_instances.append(avg_h)

        if len(masks) == 0:
            masks_tensor = torch.zeros((0, img_tensor.shape[1], img_tensor.shape[2]), dtype=torch.float32)
            classes_tensor = torch.zeros(0, dtype=torch.long)
            height_instances_tensor = torch.zeros((0, 1), dtype=torch.float32)
        else:
            masks_tensor = torch.tensor(np.stack(masks), dtype=torch.float32)
            classes_tensor = torch.tensor(classes, dtype=torch.long)
            height_instances_tensor = torch.tensor(height_instances, dtype=torch.float32).unsqueeze(1)

        output = {
            "images": img_tensor,
            "masks": masks_tensor,
            "labels": classes_tensor,
            "height_instances": height_instances_tensor,
            "height_map": torch.from_numpy(height_data_aug).unsqueeze(0).float()
        }

        return output


def collate_fn(batch):
    images = torch.stack([item['images'] for item in batch])
    targets = []
    for item in batch:
        target_dict = {
            "masks": item['masks'],
            "labels": item['labels'],
            "height_instances": item['height_instances'],
            "height_map": item['height_map']
        }
        targets.append(target_dict)
    return images, targets