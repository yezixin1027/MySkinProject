"""ISIC-2018 皮肤镜数据集封装"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import os
import cv2
import numpy as np

from .registry import register_dataset
from .preprocess import MedicalImageProcessor
from .augmentation import SegmentationAugmentation


@register_dataset("ISIC2018")
class ISIC2018Dataset(Dataset):
    """ISIC 2018 皮肤病灶分割数据集

    配置结构 (dataset 段):
        name: "ISIC2018"
        params:
          img_size: [224, 224]
          val_split: 0.2
        paths:
          train_images: "./dataset/train/images"
          train_masks: "./dataset/train/masks"
        augmentation:
          enabled: true
          flip_prob: 0.5
          rotate_prob: 0.5
          rotate_deg: 15
          brightness_prob: 0.5
    """

    def __init__(self, img_dir, mask_dir, img_size=(224, 224), is_train=True,
                 use_augmentation=True, aug_config=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = tuple(img_size)
        self.is_train = is_train

        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"找不到路径: {img_dir}")

        self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
        self.processor = MedicalImageProcessor()

        # 在线增强
        if is_train and use_augmentation:
            if aug_config:
                self.augment = SegmentationAugmentation.from_config(
                    {"augmentation": aug_config})
            else:
                self.augment = SegmentationAugmentation()
        else:
            self.augment = None

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]

        # 路径拼接
        img_path = os.path.join(self.img_dir, img_name)
        mask_name = img_name.replace(".jpg", "_segmentation.png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 读取
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"警告: 读取失败 {img_name}，跳过...")
            new_idx = (idx + 1) % len(self.img_names)
            return self.__getitem__(new_idx)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 预处理
        image = self.processor.remove_hair(image)
        if self.is_train:
            image = self.processor.apply_clahe(image)

        # Resize
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # 转 Tensor + 归一化
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        mask_tensor = (mask_tensor > 0.5).float()

        # 在线增强
        if self.augment is not None:
            image_tensor, mask_tensor = self.augment(image_tensor, mask_tensor)

        return image_tensor, mask_tensor

    # ---- 工厂方法 ----

    @classmethod
    def from_config(cls, config: dict, split: str = "train"):
        """从配置字典构建数据集 (含 train/val split)

        split: "train" → 返回训练 DataLoader
                "val"   → 返回验证 DataLoader
                "all"   → 返回完整训练集 Dataset
        """
        ds_cfg = config["dataset"]
        paths = ds_cfg["paths"]
        params = ds_cfg.get("params", {})
        aug_cfg = ds_cfg.get("augmentation", {})
        img_size = tuple(params.get("img_size", [224, 224]))

        # 解析路径 (支持相对于项目根目录)
        project_root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        train_img = _resolve_path(paths["train_images"], project_root)
        train_mask = _resolve_path(paths["train_masks"], project_root)

        if split == "all":
            return cls(train_img, train_mask, img_size=img_size,
                       is_train=True, use_augmentation=False)

        # 训练模式：完整数据集 + 在线增强
        full_ds = cls(train_img, train_mask, img_size=img_size,
                      is_train=True, use_augmentation=True,
                      aug_config=aug_cfg)

        val_split = params.get("val_split", 0.2)
        n_total = len(full_ds)
        n_train = int((1 - val_split) * n_total)
        n_val = n_total - n_train

        seed = config.get("experiment", {}).get("seed", 42)
        gen = torch.Generator().manual_seed(seed)
        train_subset, val_subset = random_split(full_ds, [n_train, n_val], generator=gen)

        if split == "train":
            return train_subset

        if split == "val":
            # 验证集不需要增强 + 不需要 CLAHE
            eval_ds = cls(train_img, train_mask, img_size=img_size,
                          is_train=False, use_augmentation=False)
            return Subset(eval_ds, val_subset.indices)

        raise ValueError(f"未知 split 参数: {split}。可选: 'train', 'val', 'all'")


def _resolve_path(path: str, root: str) -> str:
    """解析相对路径 → 绝对路径"""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(root, path))
