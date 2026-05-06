import matplotlib.pyplot as plt
import os


class TrainingVisualizer:
    def __init__(self, save_path="learning_curve.png"):
        self.save_path = save_path

    def draw(self, logs):
        epochs = [item['epoch'] for item in logs]
        losses = [item['train_loss'] for item in logs]
        dices = [item['val_dice'] for item in logs]
        ious = [item.get('val_iou', 0) for item in logs]

        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        # 绘制红色 Loss (左轴)
        ax1.plot(epochs, losses, 'r-o', label='Train Loss', markersize=4)
        ax1.set_ylabel('Loss', color='r')

        # 绘制蓝色 Dice 和 绿色 IoU (右轴)
        ax2.plot(epochs, dices, 'b-s', label='Val Dice', markersize=4)
        ax2.plot(epochs, ious, 'g-d', label='Val IoU', markersize=4)
        ax2.set_ylabel('Metrics Score', color='b')

        plt.title('Training Monitor (Real-time)')
        ax1.grid(True, linestyle='--', alpha=0.5)

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.savefig(self.save_path, dpi=150)
        plt.close()  # 必须关闭，否则多轮训练会内存溢出