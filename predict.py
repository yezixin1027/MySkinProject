"""
Res-CoordUNet 推理 & 可视化 — 使用注册器加载模型

用法:
    python predict.py                           # 使用默认配置和权重
    python predict.py --config config/xxx.yaml  # 使用自定义配置
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.utils.config import ConfigLoader
from src.models.registry import build_model
from src.dataset.registry import build_dataset
from src.training.metrics import SegmentationMetric


def main():
    parser = argparse.ArgumentParser(description="Res-CoordUNet 推理")
    parser.add_argument("--config", type=str, default="./config/default.yaml",
                        help="YAML 配置文件路径")
    parser.add_argument("--weights", type=str, default="./weights/best_model.pth",
                        help="模型权重路径")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="可视化样本数量")
    parser.add_argument("--save_dir", type=str, default="./results/visual_comparison",
                        help="结果保存目录")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 加载配置 & 模型
    cfg = ConfigLoader(args.config) if os.path.exists(args.config) else None

    model = build_model(cfg.cfg) if cfg else ResCoordUNet()
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    model.eval()

    # 2. 加载数据
    if cfg:
        val_ds = build_dataset(cfg.cfg, split="all")
    else:
        from src.dataset.isic_dataset import ISIC2018Dataset
        val_ds = ISIC2018Dataset("./data/train/images", "./data/train/masks",
                                 is_train=False, use_augmentation=False)

    metric_tool = SegmentationMetric()
    indices = np.random.choice(len(val_ds), min(args.num_samples, len(val_ds)), replace=False)

    print(f"🖼️  正在生成对比图 → {args.save_dir}")

    for idx in indices:
        image_tensor, mask_tensor = val_ds[idx]

        # 推理
        input_tensor = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)["out"]
            pred_mask = torch.sigmoid(output)

        metrics = metric_tool.calculate_all(pred_mask, mask_tensor.to(device))

        # 转为 numpy 用于绘图
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        gt_np = mask_tensor.squeeze().cpu().numpy()
        pred_np = (pred_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        # Overlay
        overlay = img_np.copy()
        overlay[pred_np == 1] = [0, 0, 255]
        combined = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)

        # 拼图
        plt.figure(figsize=(16, 4))
        display_list = [img_np, gt_np, pred_np, combined]
        titles = ['Original', 'Ground Truth',
                  f'Prediction\n(Dice:{metrics["Dice"]:.3f})', 'Overlay']

        for i in range(4):
            plt.subplot(1, 4, i + 1)
            if i in (1, 2):
                plt.imshow(display_list[i], cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(display_list[i], cv2.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, f"sample_{idx}.png"), dpi=200)
        plt.close()

    print(f"✅ 可视化完成! 共 {len(indices)} 张对比图")


if __name__ == "__main__":
    main()
