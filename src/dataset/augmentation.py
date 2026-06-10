"""在线数据增强：成对空间变换 + 颜色扰动"""

import torch
import torchvision.transforms.functional as TF
import random


class SegmentationAugmentation:
    """成对数据增强：空间变换对 图像 和 掩码 施加完全一致的操作"""

    def __init__(self, p_flip=0.5, p_rotate=0.5, p_brightness=0.5, rot_deg=15):
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.p_brightness = p_brightness
        self.rot_deg = rot_deg

    def __call__(self, image, mask):
        # image: Tensor [C, H, W]; mask: Tensor [1, H, W]

        # --- 水平翻转 ---
        if random.random() < self.p_flip:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # --- 垂直翻转 ---
        if random.random() < self.p_flip:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # --- 随机旋转 (±rot_deg 内均匀采样) ---
        if random.random() < self.p_rotate:
            angle = random.uniform(-self.rot_deg, self.rot_deg)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        # --- 亮度/对比度扰动 (仅作用于图像) ---
        if random.random() < self.p_brightness:
            brightness = random.uniform(0.85, 1.15)
            image = TF.adjust_brightness(image, brightness)
        if random.random() < self.p_brightness:
            contrast = random.uniform(0.85, 1.15)
            image = TF.adjust_contrast(image, contrast)

        return image, mask

    @classmethod
    def from_config(cls, cfg: dict):
        """从配置字典构建增强器"""
        aug_cfg = cfg.get("augmentation", {})
        if not aug_cfg.get("enabled", True):
            return None
        return cls(
            p_flip=aug_cfg.get("flip_prob", 0.5),
            p_rotate=aug_cfg.get("rotate_prob", 0.5),
            p_brightness=aug_cfg.get("brightness_prob", 0.5),
            rot_deg=aug_cfg.get("rotate_deg", 15),
        )
