"""
Res-CoordUNet 训练入口 — 配置驱动，模块化解耦

用法:
    python train.py                          # 使用 config/default.yaml
    python train.py --config config/xxx.yaml # 使用自定义配置
"""

import os
import sys
import argparse

# 确保项目根目录在 path 中
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.utils.config import ConfigLoader
from src.models.registry import build_model, list_models
from src.data.registry import build_dataset, list_datasets
from src.training.losses import build_loss, list_losses
from src.training.trainer import Trainer, build_optimizer, build_scheduler


def main():
    parser = argparse.ArgumentParser(description="Res-CoordUNet 训练")
    parser.add_argument("--config", type=str, default="./config/default.yaml",
                        help="YAML 配置文件路径")
    args = parser.parse_args()

    # 1. 加载配置
    cfg = ConfigLoader(args.config)
    print(f"[Config] {args.config}")
    print(f"   可用模型: {list_models()}")
    print(f"   可用数据集: {list_datasets()}")
    print(f"   可用损失函数: {list_losses()}")

    # 2. 构建数据管道
    train_loader_cfg = cfg.cfg
    batch_size = cfg.get("training.batch_size", 12)
    num_workers = cfg.get("training.num_workers", 8)
    use_cuda = torch_available()

    train_ds = build_dataset(cfg.cfg, split="train")
    val_ds = build_dataset(cfg.cfg, split="val")

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_cuda)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=use_cuda)

    # 3. 构建模型
    model = build_model(cfg.cfg)

    # 4. 构建损失函数 + 优化器 + 调度器
    loss_fn = build_loss(cfg.cfg)
    optimizer = build_optimizer(model, cfg.cfg)
    scheduler = build_scheduler(optimizer, cfg.cfg, cfg.get("training.epochs", 200))

    # 5. 启动训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=cfg.cfg,
    )
    trainer.train()


def torch_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
