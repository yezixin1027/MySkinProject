"""训练引擎：组织训练/验证循环，触发回调生命周期，管理 AMP 和设备切换"""

import os
import sys
import time
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any

from .metrics import SegmentationMetric
from .callbacks import build_callbacks, Callback


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    """从配置构建优化器"""
    opt_cfg = config["training"]["optimizer"]
    name = opt_cfg["name"]
    params = opt_cfg.get("params", {})

    if name == "AdamW":
        return optim.AdamW(model.parameters(), **params)
    elif name == "Adam":
        return optim.Adam(model.parameters(), **params)
    elif name == "SGD":
        return optim.SGD(model.parameters(), **params)
    else:
        raise KeyError(f"未知优化器: {name}。可选: AdamW, Adam, SGD")


def build_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any], epochs: int):
    """从配置构建学习率调度器"""
    sched_cfg = config["training"].get("scheduler")
    if sched_cfg is None:
        return None

    name = sched_cfg["name"]
    params = sched_cfg.get("params", {})

    if name == "WarmupCosine":
        warmup_epochs = params.get("warmup_epochs", 5)
        min_lr_factor = params.get("min_lr", 1e-6)

        def lr_lambda(current_epoch):
            if current_epoch < warmup_epochs:
                return float(current_epoch) / float(max(1, warmup_epochs))
            progress = float(current_epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise KeyError(f"未知调度器: {name}。可选: WarmupCosine")


class Trainer:
    """
    训练引擎：解耦训练逻辑与辅助功能（日志、早停、可视化）

    使用方式:
        trainer = Trainer(model, train_loader, val_loader, loss_fn, optimizer, scheduler, config)
        trainer.train()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        config: Dict[str, Any],
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config

        # 硬件
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.use_amp = torch.cuda.is_available() and config.get("training", {}).get("amp", {}).get("enabled", True)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # 配置项
        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.metric_tool = SegmentationMetric()

        # 回调系统
        self.callbacks = build_callbacks(config, scheduler=scheduler)

        # 内部状态
        self._stop_early = False
        self.logs_history = []

    def train(self):
        """启动完整训练流程"""
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        print(f"[Train] 设备: {gpu_name}")
        print(f"Train: {len(self.train_loader.dataset)} samples | Val: {len(self.val_loader.dataset)} samples")

        self._fire("on_train_start")

        for epoch in range(self.epochs):
            self._fire("on_epoch_start", epoch)

            # A. 训练
            self.model.train()
            accu_loss = 0.0
            start_time = time.time()
            current_lr = self.optimizer.param_groups[0]['lr']

            train_bar = tqdm(self.train_loader, file=sys.stdout, colour='green')
            train_bar.set_description(f"Epoch [{epoch + 1}/{self.epochs}]")

            for i, (images, masks) in enumerate(train_bar):
                images, masks = images.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(images)["out"]
                        loss = self.loss_fn(outputs, masks)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)["out"]
                    loss = self.loss_fn(outputs, masks)
                    loss.backward()
                    self.optimizer.step()

                accu_loss += loss.item()
                train_bar.set_postfix({
                    "loss": f"{accu_loss / (i + 1):.4f}",
                    "lr": f"{current_lr:.2e}"
                })

            train_loss = accu_loss / len(self.train_loader)

            # B. 验证
            self.model.eval()
            val_metrics = {"Dice": 0.0, "IoU": 0.0, "Sens": 0.0, "Spec": 0.0}

            with torch.no_grad():
                for v_images, v_masks in self.val_loader:
                    v_images, v_masks = v_images.to(self.device), v_masks.to(self.device)
                    v_outputs = self.model(v_images)["out"]
                    batch_res = self.metric_tool.calculate_all(v_outputs, v_masks)
                    for k in val_metrics.keys():
                        val_metrics[k] += batch_res[k]

            num_val = len(self.val_loader)
            avg_metrics = {k: v / num_val for k, v in val_metrics.items()}
            epoch_time = time.time() - start_time

            # 组装日志
            logs = {
                "train_loss": train_loss,
                "epoch_time": epoch_time,
                "lr": f"{current_lr:.8f}",
                **avg_metrics
            }

            # C. 触发回调
            self._fire("on_epoch_end", epoch, logs)

            if self._stop_early:
                break

        self._fire("on_train_end")
        print(f"Training complete!")

    def _fire(self, hook: str, *args):
        """触发所有回调的指定钩子"""
        for cb in self.callbacks:
            getattr(cb, hook)(self, *args)
