import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.build_model import ResCoordUNet
from data_utils.dataset import ISIC2018Dataset
from train_utils.metrics import SegmentationMetric


def main():
    # 1. 基础配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = "./weights/best_model.pth"
    save_dir = "./results/visual_comparison"
    os.makedirs(save_dir, exist_ok=True)

    # 2. 加载模型与权重
    model = ResCoordUNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 3. 加载验证集 (这里用 val_ds 抽样展示)
    val_ds = ISIC2018Dataset("./data/val/images", "./data/val/masks", is_train=False)
    metric_tool = SegmentationMetric()

    # 4. 随机抽取样本进行预测 (比如抽 10 张)
    num_samples = 10
    indices = np.random.choice(len(val_ds), num_samples, replace=False)

    print(f"🖼️ 正在生成论文对比图，保存至: {save_dir}")

    for idx in indices:
        image_tensor, mask_tensor = val_ds[idx]
        # image_tensor: [3, H, W], mask_tensor: [1, H, W]

        # 模型推理
        input_tensor = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)["out"]
            pred_mask = torch.sigmoid(output)

        # 计算单张图的指标
        metrics = metric_tool.calculate_all(pred_mask, mask_tensor.to(device))

        # 数据转换用于绘图
        # 原图还原 (假设做了归一化，这里转回 0-255)
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # 注意颜色通道

        gt_np = mask_tensor.squeeze().cpu().numpy()
        pred_np = (pred_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        # 生成 Overlay (半透明红色覆盖在病灶上)
        overlay = img_np.copy()
        overlay[pred_np == 1] = [0, 0, 255]  # BGR 红色
        combined_view = cv2.addWeighted(img_np, 0.7, overlay, 0.3, 0)

        # 5. 使用 Matplotlib 拼图
        plt.figure(figsize=(16, 4))

        display_list = [img_np, gt_np, pred_np, combined_view]
        titles = ['Original Image', 'Ground Truth',
                  f'Prediction\n(Dice:{metrics["Dice"]:.3f})',
                  'Overlay Result']

        for i in range(4):
            plt.subplot(1, 4, i + 1)
            if i == 1 or i == 2:
                plt.imshow(display_list[i], cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(display_list[i], cv2.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sample_{idx}.png"), dpi=200)
        plt.close()

    print(f"✅ 可视化完成！快去查看 {save_dir} 里的对比图吧。")


if __name__ == "__main__":
    main()