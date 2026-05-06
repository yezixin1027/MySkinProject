# 基于 Res-CoordUNet 的 ISIC-2018 皮肤病灶分割系统

> **项目状态**：训练中 / 实验阶段  
> **核心架构**：Residual Learning + Coordinate Gated Attention + Hybrid Loss

---

## 🛠️ 训练技巧与超参数 (Training Highlights)

### 核心训练算法
* **混合精度加速 (AMP)**：通过 `torch.amp` 自动切换精度。针对 **RTX 4050 (6GB)** 深度优化，显存占用降低约 40%，计算速度提升近 2 倍。
* **带预热的余弦退火 (Warmup + Cosine Annealing)**：
  - **Warmup Stage**: 前 5 个 Epoch 学习率由 $10^{-6}$ 线性增长至 $10^{-4}$，确保随机初始化后的梯度稳定性。
  - **Annealing Stage**: 预热结束后进入余弦衰减模式，使模型平滑地滑向全局最优解。
* **早停机制 (Early Stopping)**：设定 `Patience=10`，若连续 10 轮验证集指标未提升则自动停止训练，防止过拟合。
* **混合损失函数 (Hybrid Loss)**：结合 **Dice Loss** 与 **Focal Loss**，有效缓解背景区域远大于病灶区域导致的样本极度不平衡问题。
---

## 📑 目录
1. [项目简介](#1-项目简介)
2. [模型核心创新](#2-模型核心创新)
3. [项目目录结构](#3-项目目录结构)
4. [算法优势对比](#4-算法优势对比)
5. [环境要求与运行](#5-环境要求与运行)
6. [实验结果](#6-实验结果)

## 🚀 1. 项目简介
本项目旨在利用深度学习技术对 ISIC-2018（Task 1）挑战赛提供的皮肤镜图像进行自动化分割。针对皮肤病灶边缘模糊、毛发伪影严重及对比度低等痛点，本项目在经典 U-Net 基础上进行了深度改进，提出并实现了 **Res-CoordUNet** 模型。

---

## 💡 2. 模型核心创新 (Core Innovations)

### 2.1 残差学习模块 (Residual Learning)
* **技术逻辑**：采用 **Residual Block** 替换传统卷积层。
* **优势**：通过“快捷路径（Shortcut）”保护原始图像细节，有效防止深层网络的梯度消失，使模型对病灶边缘的捕捉更细腻，收敛速度明显优于传统卷积。

### 2.2 坐标门控注意力 (Coordinate Gated Attention, CGA)
* **技术逻辑**：在跳跃连接（Skip Connection）处集成坐标感知与语义门控。
* **优势**：
    * **位置感知**：利用水平和垂直方向的平均池化，使模型具备感知病灶“经纬度”的能力。
    * **噪声抑制**：通过深层特征作为门控信号，自动过滤皮肤纹理、折痕和毛发干扰，显著降低误报率。

### 2.3 混合损失函数 (Hybrid Loss)
* **技术逻辑**：结合 **Dice Loss**（关注区域重叠度）与 **Focal Loss**（关注难区分像素）。
* **优势**：有效缓解了背景区域远大于病灶区域导致的样本极度不平衡问题，强制网络学习边缘细节。

---

## 📂 3. 项目目录结构
```text
MySkinProject/
├── src/                   # 🧠 核心算法层
│   ├── unet_parts.py      # 基础组件：升级为 ResidualBlock (残差块)
│   ├── attention_gate.py  # 创新模块：Coordinate Gated Attention (坐标门控)
│   └── build_model.py     # 组装类：构建完整的 Res-CoordUNet 架构
├── data_utils/            # 📦 数据处理层
│   ├── preprocess.py      # 预处理：DullRazor 去毛发、CLAHE 增强
│   └── dataset.py         # 自定义 Dataset：支持实时数据增强
├── train_utils/           # 🛠️ 训练辅助层
│   ├── losses.py          # 混合损失：Dice + Focal Loss 实现
│   ├── metrics.py         # 评估指标：Dice, IoU, Sensitivity 等
│   └── distributed_utils.py # 训练通用工具函数
├── weights/               # 💾 权重存档点：保存最优模型 (.pth)
├── train.py               # 🚀 训练主脚本 (集成 AMP 加速、余弦退火、进度条)
├── predict.py             # 🔍 结果预测与可视化脚本
├── requirements.txt       # 📋 环境依赖清单
└── README.md              # 📝 项目说明文档