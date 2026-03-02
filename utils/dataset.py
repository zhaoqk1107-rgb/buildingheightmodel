import os
import sys
from glob import glob
import numpy as np
import rasterio
import warnings

warnings.filterwarnings('ignore')
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RSDataset(Dataset):
    def __init__(self, dataset_names, num_sample=-1, is_train=True):
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
        return len(self.img_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.label_paths[idx]) as src:
            height_data = src.read(1).astype(np.float32)

        valid_mask = ~(np.isnan(height_data) | np.isinf(height_data) | (height_data < 0) | (height_data == 255) | (
                    height_data > 1000))
        height_data = np.where(valid_mask, height_data, 0)

        with rasterio.open(self.img_paths[idx]) as src:
            img_data = np.stack([src.read(i) for i in range(1, 4)])

        img_data = np.transpose(img_data, (1, 2, 0))
        augmented = self.transform(image=img_data, mask=height_data)
        img_tensor = augmented['image']
        height_data_aug = augmented['mask']

        if isinstance(height_data_aug, torch.Tensor):
            height_data_aug = height_data_aug.numpy()

        label_binary = (height_data_aug > 0).astype(np.uint8)

        # 为了应对粘连，可以在生成实例标签时进行一次轻微腐蚀操作
        # 这样网络在学习实例中心时会更加聚焦，推理后再由边界损失恢复形状
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        label_eroded = cv2.erode(label_binary, kernel, iterations=1)
        num_objs, labels_im = cv2.connectedComponents(label_eroded)

        masks = []
        classes = []
        height_instances = []
        for i in range(1, num_objs):
            mask_loc = (labels_im == i)
            if mask_loc.sum() > 1:
                mask_i = mask_loc.astype(np.float32)
                masks.append(mask_i)
                classes.append(0)

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