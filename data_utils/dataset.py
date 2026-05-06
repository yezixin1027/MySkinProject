import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np


try:
    from .preprocess import MedicalImageProcessor
except ImportError:
    from preprocess import MedicalImageProcessor


class ISIC2018Dataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(224, 224), is_train=True):
        """
        ISIC 2018 专用数据集类
        :param img_dir: 原始图片文件夹路径 (images)
        :param mask_dir: 标签图片文件夹路径 (masks)
        :param img_size: 输入模型的尺寸，默认 224x224
        :param is_train: 是否为训练模式（开启 CLAHE 增强）
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.is_train = is_train

        # 1. 只读取 .jpg 文件，彻底过滤杂质文件
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"找不到路径: {img_dir}")

        self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
        self.processor = MedicalImageProcessor()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        # 获取图片文件名
        img_name = self.img_names[idx]

        # 1. 路径拼接
        img_path = os.path.join(self.img_dir, img_name)
        # 严格对应 ISIC 命名：ISIC_0000000.jpg -> ISIC_0000000_segmentation.png
        mask_name = img_name.replace(".jpg", "_segmentation.png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 2. 读取图片
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 【鲁棒性检查】：防止读取失败导致程序崩溃
        if image is None or mask is None:
            print(f"⚠️ 警告: 读取失败 {img_name}，正在跳过...")
            new_idx = (idx + 1) % len(self.img_names)
            return self.__getitem__(new_idx)

        # OpenCV 默认 BGR，需转为 RGB 以符合生理特征和模型习惯
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 3. 调用 MedicalImageProcessor 预处理
        # 刮毛是必须的，保持特征纯净
        image = self.processor.remove_hair(image)

        # 只有训练集才开启对比度增强（增加样本多样性）
        if self.is_train:
            image = self.processor.apply_clahe(image)

        # 4. 尺寸调整 (Resize)
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)  # 标签用最近邻插值，防止产生模糊权重

        # 5. 格式转换与归一化
        # [H, W, C] -> [C, H, W]，并缩放到 [0.0, 1.0]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Mask 增加通道维 [H, W] -> [1, H, W]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        # 二值化：确保标签只有 0 和 1（去除插值产生的灰度值）
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor


# --- 测试脚本  ---
if __name__ == "__main__":
    # 1. 自动获取当前文件所在的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 2. 获取 data_utils 的上级目录 (即项目根目录 MySkinProject)
    project_root = os.path.dirname(os.path.dirname(current_file_path))

    # 3. 拼接出正确的 data 路径
    test_img_dir = os.path.join(project_root, "data", "train", "images")
    test_mask_dir = os.path.join(project_root, "data", "train", "masks")

    print(f"🔍 正在检查路径: {test_img_dir}")

    if os.path.exists(test_img_dir):
        train_ds = ISIC2018Dataset(img_dir=test_img_dir, mask_dir=test_mask_dir)
        print(f"✅ 数据管道通畅！")
        print(f"   项目根目录: {project_root}")
        print(f"   样本总量: {len(train_ds)}")
        if len(train_ds) > 0:
            img, label = train_ds[0]
            print(f"   图像维度: {img.shape}")
            print(f"   标签维度: {label.shape}")
            print(f"   标签值域: {torch.unique(label)}")
    else:
        print("❌ 路径依然不对！")
        print(f"   请确认你的 MySkinProject 文件夹下是否有 data 文件夹。")
        print(f"   当前脚本尝试访问的绝对路径是: {os.path.abspath(test_img_dir)}")