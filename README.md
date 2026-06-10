# Res-CoordUNet: ISIC-2018 皮肤病灶语义分割

> **架构**: YAML 配置驱动 | 注册器模式 | 回调系统  
> **模型**: Residual U-Net + Coordinate Gated Attention  
> **框架**: PyTorch

---

## 架构概览

本项目采用 **YAML 配置驱动的模块化架构**，将数据集、模型、训练器、可视化四大模块完全解耦。

```
config/default.yaml        ← 所有超参数一处管理，无需改代码
       │
       ▼
src/
├── models/                ← 模型注册器 → build_model(config)
│   ├── registry.py           @register_model("ResCoordUNet")
│   ├── blocks.py             ResidualBlock / Down / OutConv
│   ├── attention.py          CoordGatedAttention (CGA)
│   └── res_coord_unet.py     Up + ResCoordUNet
├── data/                  ← 数据集注册器 → build_dataset(config, split)
│   ├── registry.py           @register_dataset("ISIC2018")
│   ├── preprocess.py         DullRazor + CLAHE + GrayWorld
│   ├── augmentation.py       SegmentationAugmentation
│   └── isic_dataset.py       ISIC2018Dataset + from_config()
├── training/              ← 训练引擎 + 回调系统
│   ├── losses.py             DiceLoss / FocalLoss / HybridLoss + LOSS_REGISTRY
│   ├── metrics.py            Dice, IoU, Sensitivity, Specificity
│   ├── trainer.py            Trainer (AMP, device, epoch loop)
│   └── callbacks.py          EarlyStopping / Checkpoint / Viz / Log / LR
└── utils/
    ├── config.py             ConfigLoader (YAML → dict + 校验 + 点号访问)
    └── visualizer.py         TrainingVisualizer
```

**入口脚本**（极简，~30 行）：

```python
# train.py
cfg = ConfigLoader("./config/default.yaml")
train_ds = build_dataset(cfg.cfg, split="train")
val_ds   = build_dataset(cfg.cfg, split="val")
model    = build_model(cfg.cfg)
loss_fn  = build_loss(cfg.cfg)
Trainer(model, train_loader, val_loader, loss_fn, optimizer, scheduler, cfg.cfg).train()
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练（使用默认配置）
python train.py

# 3. 使用自定义配置
python train.py --config config/your_experiment.yaml

# 4. 推理 & 可视化
python predict.py --weights ./weights/best_model.pth --num_samples 10
```

---

## YAML 配置

所有超参数集中在 `config/default.yaml`，支持配置继承和覆盖：

```yaml
experiment:
  name: "res_coord_unet_isic2018"
  seed: 42

model:
  name: "ResCoordUNet"          # 换模型只需改这一行
  params:
    in_channels: 3
    num_classes: 1
    bilinear: true
    base_c: 64

dataset:
  name: "ISIC2018"              # 换数据集只需改这一行
  params:
    img_size: [224, 224]
    val_split: 0.2              # 80/20 划分，扩大验证集
  augmentation:
    enabled: true
    flip_prob: 0.5
    rotate_prob: 0.5
    rotate_deg: 15

training:
  batch_size: 12
  epochs: 200
  loss:
    name: "HybridLoss"          # 换损失只需改这一行
    params: {dice_weight: 0.6, focal_weight: 0.4}
  optimizer:
    name: "AdamW"
    params: {lr: 0.0001, weight_decay: 0.0001}
  scheduler:
    name: "WarmupCosine"
    params: {warmup_epochs: 5, min_lr: 0.000001}
  callbacks:
    early_stopping:  {enabled: true, monitor: "val_dice", patience: 20}
    model_checkpoint: {enabled: true, save_dir: "./weights"}
    visualization:    {enabled: true, save_path: "./learning_curve.png"}
    logging:          {enabled: true, log_path: "./train_results.json"}
```

---

## 模型核心创新

### 残差学习 (ResidualBlock)

双卷积路径 + 1×1 shortcut，保护梯度流动，加速收敛。

### 坐标门控注意力 (CoordGatedAttention)

| 分支 | 机制 | 作用 |
|------|------|------|
| 坐标感知 | H/W 方向池化 → 1×1 位置编码 → Sigmoid | 赋予模型"经纬度"位置感知 |
| 语义门控 | 深层特征 → 上采样 → 1×1 → Sigmoid | 抑制皮肤纹理、毛发噪声 |

最终输出 = `x ⊙ attention_coord ⊙ attention_gate`

### 混合损失 (Dice + Focal)

Dice Loss 关注区域重叠度，Focal Loss 聚焦困难边缘像素，有效缓解正负样本极度不平衡。

---

## 训练策略

| 技术 | 说明 |
|------|------|
| AMP 混合精度 | RTX 4050 显存降低 ~40%，训练提速 ~2× |
| Warmup + Cosine Annealing | 前 5 epoch 从 1e-6 升至 1e-4，之后余弦衰减 |
| 早停 | patience=20，监控 val_dice |
| 在线增强 | 随机翻转 / 旋转 / 亮度扰动，空间变换对 image-mask 同步 |

---

## 数据预处理流水线

```
原图 → DullRazor 去毛发 → 灰度世界校色 → CLAHE 对比度增强 → 模型输入
```

- **DullRazor**: 黑帽运算提取毛发 → 阈值二值化 → Telea Inpainting 修复
- **GrayWorld**: 按通道均值比例修正 RGB，消除光照偏差
- **CLAHE**: LAB 空间 L 通道自适应直方图均衡化 (clipLimit=2.0, tile=8×8)

---

## 实验结果 (48 Epoch, RTX 4050 6GB)

| 指标 | 数值 |
|------|------|
| Dice | 0.854 |
| IoU | 0.766 |
| Sensitivity | 0.903 |
| Specificity | 0.969 |

---

## 扩展指南

### 添加新模型

```python
from src.models.registry import register_model

@register_model("MyUNet")
class MyUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, **kwargs):
        ...
```

然后在 YAML 中 `model.name: "MyUNet"` 即可使用。

### 添加新数据集

```python
from src.data.registry import register_dataset

@register_dataset("MyDataset")
class MyDataset(Dataset):
    @classmethod
    def from_config(cls, config, split):
        ...
```

### 添加自定义回调

```python
from src.training.callbacks import Callback

class MyCallback(Callback):
    def on_epoch_end(self, trainer, epoch, logs):
        ...
```

---

## 目录结构

```
MySkinProject/
├── config/default.yaml        # YAML 配置
├── src/                       # 核心代码
│   ├── models/                # 模型层 + 注册器
│   ├── data/                  # 数据层 + 注册器
│   ├── training/              # 训练引擎 + 回调 + 损失 + 指标
│   └── utils/                 # 配置加载 + 可视化
├── data/                      # ISIC-2018 数据集
│   ├── train/images/          # 2576 张训练图
│   └── train/masks/           # 2596 个标注
├── weights/                   # 模型权重
├── train.py                   # 训练入口
├── predict.py                 # 推理入口
└── requirements.txt
```

---

## 依赖

- Python >= 3.10
- PyTorch >= 2.0
- torchvision, opencv-python, numpy, matplotlib, tqdm, scipy, scikit-image, pyyaml
