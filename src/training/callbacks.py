"""
训练回调系统：将日志、早停、checkpoint、可视化等功能从训练循环中解耦。

所有回调继承 Callback 基类，按生命周期钩子被 Trainer 调用:
    on_train_start → on_epoch_start → on_epoch_end → on_train_end
"""

import os
import json
import torch
from typing import Dict, Any, List, Optional


class Callback:
    """回调基类"""
    def on_train_start(self, trainer): pass
    def on_epoch_start(self, trainer, epoch: int): pass
    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]): pass
    def on_train_end(self, trainer): pass


class EarlyStoppingCallback(Callback):
    """早停回调：监控指定指标，patience 轮未提升则停止训练"""

    def __init__(self, monitor: str = "val_dice", patience: int = 20,
                 mode: str = "max", min_delta: float = 0.0):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        current = logs.get(self.monitor)
        if current is None:
            return

        if self.best_score is None:
            self.best_score = current
            self.best_weights = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}
            return

        improved = (current > self.best_score + self.min_delta) if self.mode == "max" \
              else (current < self.best_score - self.min_delta)

        if improved:
            self.best_score = current
            self.counter = 0
            self.best_weights = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer._stop_early = True
                # 恢复最佳权重
                if self.best_weights is not None:
                    trainer.model.load_state_dict(self.best_weights)
                print(f"[EarlyStopping] Triggered! Best {self.monitor}: {self.best_score:.4f}")


class ModelCheckpointCallback(Callback):
    """模型检查点回调：保存最佳或定期 checkpoint"""

    def __init__(self, monitor: str = "val_dice", mode: str = "max",
                 save_dir: str = "./weights", save_best_only: bool = True):
        self.monitor = monitor
        self.mode = mode
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.best_score = None
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        current = logs.get(self.monitor)
        if current is None:
            return

        is_best = False
        if self.best_score is None or (
            (self.mode == "max" and current > self.best_score) or
            (self.mode == "min" and current < self.best_score)
        ):
            self.best_score = current
            is_best = True

        if is_best or not self.save_best_only:
            path = os.path.join(self.save_dir,
                                f"best_model.pth" if is_best else f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(trainer.model.state_dict(), path)
            if is_best:
                print(f"[Checkpoint] New best model saved (epoch {epoch+1}, {self.monitor}={current:.4f})")


class LoggingCallback(Callback):
    """日志回调：将每轮训练指标写入 JSON 文件"""

    def __init__(self, log_path: str = "./train_results.json"):
        self.log_path = log_path
        self.logs: List[Dict] = []

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        self.logs.append({
            "epoch": epoch + 1,
            "train_loss": round(logs.get("train_loss", 0), 4),
            "val_dice": round(logs.get("Dice", 0), 4),
            "val_iou": round(logs.get("IoU", 0), 4),
            "val_sens": round(logs.get("Sens", 0), 4),
            "val_spec": round(logs.get("Spec", 0), 4),
            "lr": logs.get("lr", ""),
            "time": f"{logs.get('epoch_time', 0):.1f}s"
        })
        with open(self.log_path, "w") as f:
            json.dump(self.logs, f, indent=4)

    def on_train_end(self, trainer):
        with open(self.log_path, "w") as f:
            json.dump(self.logs, f, indent=4)
        print(f"[Log] Training log saved to {self.log_path}")


class VisualizationCallback(Callback):
    """可视化回调：每轮实时绘制学习曲线"""

    def __init__(self, save_path: str = "./learning_curve.png"):
        self.save_path = save_path
        self.epochs: List[int] = []
        self.losses: List[float] = []
        self.dices: List[float] = []
        self.ious: List[float] = []

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        self.epochs.append(epoch + 1)
        self.losses.append(logs.get("train_loss", 0))
        self.dices.append(logs.get("Dice", 0))
        self.ious.append(logs.get("IoU", 0))
        self._draw()

    def _draw(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        ax1.plot(self.epochs, self.losses, 'r-o', label='Train Loss', markersize=4)
        ax1.set_ylabel('Loss', color='r')

        ax2.plot(self.epochs, self.dices, 'b-s', label='Val Dice', markersize=4)
        ax2.plot(self.epochs, self.ious, 'g-d', label='Val IoU', markersize=4)
        ax2.set_ylabel('Metrics Score', color='b')

        plt.title('Training Monitor (Real-time)')
        ax1.grid(True, linestyle='--', alpha=0.5)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.savefig(self.save_path, dpi=150)
        plt.close()


class LRSchedulerCallback(Callback):
    """学习率调度回调：每个 epoch 结束后 step scheduler"""

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_end(self, trainer, epoch: int, logs: Dict[str, Any]):
        if self.scheduler is not None:
            self.scheduler.step()


# ---------------------------------------------------------------------------
# 回调工厂
# ---------------------------------------------------------------------------

_CALLBACK_REGISTRY = {
    "early_stopping": EarlyStoppingCallback,
    "model_checkpoint": ModelCheckpointCallback,
    "visualization": VisualizationCallback,
    "logging": LoggingCallback,
}


def build_callbacks(config: Dict[str, Any], scheduler=None) -> List[Callback]:
    """从配置字典构建所有启用的回调"""
    callbacks_cfg = config.get("training", {}).get("callbacks", {})
    callbacks: List[Callback] = []

    for name, cfg in callbacks_cfg.items():
        if not isinstance(cfg, dict) or not cfg.get("enabled", True):
            continue

        if name in _CALLBACK_REGISTRY:
            # 过滤掉 'enabled' 字段
            kwargs = {k: v for k, v in cfg.items() if k != "enabled"}
            callbacks.append(_CALLBACK_REGISTRY[name](**kwargs))

    # 如果配置了 scheduler，自动添加 LRSchedulerCallback
    if scheduler is not None:
        callbacks.append(LRSchedulerCallback(scheduler))

    return callbacks
