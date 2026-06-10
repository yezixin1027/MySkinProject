"""损失函数模块：Dice Loss, Focal Loss, 混合损失 + 注册器"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_REGISTRY = {}


def register_loss(name: str):
    """装饰器：将损失函数注册到 LOSS_REGISTRY"""
    def decorator(cls):
        if name in LOSS_REGISTRY:
            raise KeyError(f"损失函数 '{name}' 已注册。")
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator


def list_losses() -> list:
    """列出所有已注册的损失函数名称"""
    return list(LOSS_REGISTRY.keys())


def build_loss(config: dict) -> nn.Module:
    """从配置字典构建损失函数"""
    loss_cfg = config["training"]["loss"]
    name = loss_cfg["name"]
    params = loss_cfg.get("params", {})

    if name not in LOSS_REGISTRY:
        available = ", ".join(LOSS_REGISTRY.keys())
        raise KeyError(f"未知损失函数 '{name}'。可用: {available}")

    return LOSS_REGISTRY[name](**params)


@register_loss("DiceLoss")
class DiceLoss(nn.Module):
    """Dice Loss: 关注区域重叠度，天然抗类别不平衡"""

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        predict = torch.sigmoid(predict)
        predict = predict.view(-1)
        target = target.view(-1)
        intersection = (predict * target).sum()
        dice = (2. * intersection + self.smooth) / (predict.sum() + target.sum() + self.smooth)
        return 1 - dice


@register_loss("FocalLoss")
class FocalLoss(nn.Module):
    """Focal Loss: 聚焦困难样本（边缘像素），降低易分样本权重"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predict, target):
        bce_loss = F.binary_cross_entropy_with_logits(predict, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


@register_loss("HybridLoss")
class HybridLoss(nn.Module):
    """混合损失: Dice + Focal 加权求和"""

    def __init__(self, dice_weight=0.6, focal_weight=0.4,
                 focal_alpha=0.25, focal_gamma=2.0):
        super(HybridLoss, self).__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, predict, target):
        loss_dice = self.dice(predict, target)
        loss_focal = self.focal(predict, target)
        return self.dice_weight * loss_dice + self.focal_weight * loss_focal
