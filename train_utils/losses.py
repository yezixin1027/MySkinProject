import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss: 专门用于处理类别不平衡，关注预测区域与真实区域的交集"""

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        predict = torch.sigmoid(predict)

        # 将预测值和标签展平
        predict = predict.view(-1)
        target = target.view(-1)

        intersection = (predict * target).sum()
        dice = (2. * intersection + self.smooth) / (predict.sum() + target.sum() + self.smooth)

        return 1 - dice


class FocalLoss(nn.Module):
    """Focal Loss: 强迫模型关注那些长得像皮肤的病灶边缘（困难样本）"""

    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predict, target):
        # 使用 BCE 作为基础
        bce_loss = F.binary_cross_entropy_with_logits(predict, target, reduction='none')
        pt = torch.exp(-bce_loss)  # 预测的置信度

        # 重点惩罚那些预测错的困难样本
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss.mean()


class HybridLoss(nn.Module):
    """三位一体混合损失：Dice + Focal"""

    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super(HybridLoss, self).__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, predict, target):
        loss_dice = self.dice(predict, target)
        loss_focal = self.focal(predict, target)

        # 加权求和
        return self.dice_weight * loss_dice + self.focal_weight * loss_focal