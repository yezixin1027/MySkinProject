import os
import sys
import json
import time
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 环境适配：确保能找到自定义模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from src.build_model import ResCoordUNet
from data_utils.dataset import ISIC2018Dataset
from train_utils.losses import HybridLoss
from train_utils.distributed_utils import evaluate
from train_utils.metrics import SegmentationMetric
from train_utils.image import TrainingVisualizer


def main():
    # ---------------------------------------------------------
    # 1. 硬件与超参数配置
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 12
    epochs = 200
    initial_lr = 1e-4
    warmup_epochs = 5
    patience = 20

    # ---------------------------------------------------------
    # 2. 数据流加载
    # ---------------------------------------------------------
    train_ds = ISIC2018Dataset("./data/train/images", "./data/train/masks", is_train=True)
    val_ds = ISIC2018Dataset("./data/val/images", "./data/val/masks", is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=8, pin_memory=True)

    # ---------------------------------------------------------
    # 3. 初始化工具类
    # ---------------------------------------------------------
    model = ResCoordUNet(in_channels=3, num_classes=1).to(device)
    criterion = HybridLoss(dice_weight=0.6, focal_weight=0.4)
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)

    # 实例化指标计算器和绘图器
    metric_tool = SegmentationMetric()
    visualizer = TrainingVisualizer(save_path="./learning_curve.png")

    # 定义带 Warmup 的学习率调度函数
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler('cuda')

    # ---------------------------------------------------------
    # 4. 训练监控变量
    # ---------------------------------------------------------
    best_dice = 0.0
    early_stop_count = 0
    train_logs = []
    if not os.path.exists("./weights"): os.makedirs("./weights")

    print(f"🔥 训练开启 | GPU: {torch.cuda.get_device_name(0)} | 样本数: {len(train_ds)}")

    # ---------------------------------------------------------
    # 5. 主循环
    # ---------------------------------------------------------
    for epoch in range(epochs):
        # --- A. 训练阶段 ---
        model.train()
        accu_loss = 0.0
        start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        train_bar = tqdm(train_loader, file=sys.stdout, colour='green')
        train_bar.set_description(f"Epoch [{epoch + 1}/{epochs}]")

        for i, (images, masks) in enumerate(train_bar):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                outputs = model(images)["out"]
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            accu_loss += loss.item()
            train_bar.set_postfix({
                "loss": f"{accu_loss / (i + 1):.4f}",
                "lr": f"{current_lr:.2e}"
            })

        # --- B. 验证阶段 (全指标计算) ---
        model.eval()
        val_metrics = {"Dice": 0.0, "IoU": 0.0, "Sens": 0.0, "Spec": 0.0}

        with torch.no_grad():
            for v_images, v_masks in val_loader:
                v_images, v_masks = v_images.to(device), v_masks.to(device)
                v_outputs = model(v_images)["out"]

                # 使用你的 SegmentationMetric 类计算 batch 指标
                batch_res = metric_tool.calculate_all(v_outputs, v_masks)
                for k in val_metrics.keys():
                    val_metrics[k] += batch_res[k]

        # 计算验证集平均值
        num_val_batches = len(val_loader)
        avg_metrics = {k: v / num_val_batches for k, v in val_metrics.items()}

        scheduler.step()  # 更新下一轮学习率
        epoch_time = time.time() - start_time
        train_loss = accu_loss / len(train_loader)

        # --- C. 日志记录与可视化 ---
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_dice": round(avg_metrics["Dice"], 4),
            "val_iou": round(avg_metrics["IoU"], 4),
            "val_sens": round(avg_metrics["Sens"], 4),
            "val_spec": round(avg_metrics["Spec"], 4),
            "lr": f"{current_lr:.8f}",
            "time": f"{epoch_time:.1f}s"
        }
        train_logs.append(log_entry)

        # 实时保存 JSON 并刷新 learning_curve.png
        with open("train_results.json", "w") as f:
            json.dump(train_logs, f, indent=4)
        visualizer.draw(train_logs)

        # --- D. 最佳模型保存与早停 ---
        current_dice = avg_metrics["Dice"]
        if current_dice > best_dice:
            best_dice = current_dice
            early_stop_count = 0
            torch.save(model.state_dict(), "./weights/best_model.pth")
            print(f"✨ 新纪录! Best Dice: {best_dice:.4f} (已同步刷新可视化看板)")
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print(f"🛑 触发早停! 最终最高 Dice: {best_dice:.4f}")
                break

    print(f"🏁 训练圆满结束!")


if __name__ == "__main__":
    main()