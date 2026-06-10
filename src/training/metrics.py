"""分割评估指标：Dice, IoU, Sensitivity, Specificity"""

import torch
import numpy as np


class SegmentationMetric:
    """图像分割指标计算器"""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def calculate_all(self, predict, target):
        """
        计算所有关键指标
        predict: 模型输出 (Tensor, logits 或 sigmoid 后均可)
        target:  真实标签 (Tensor)
        """
        # 自动检测是否为 logits
        if predict.max() > 1.0 or predict.min() < 0.0:
            predict = torch.sigmoid(predict)

        predict = (predict > self.threshold).float()
        target = (target > 0.5).float()

        predict = predict.view(-1).cpu().numpy()
        target = target.view(-1).cpu().numpy()

        tp = np.sum((predict == 1) & (target == 1))
        fp = np.sum((predict == 1) & (target == 0))
        tn = np.sum((predict == 0) & (target == 0))
        fn = np.sum((predict == 0) & (target == 1))

        dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
        iou = tp / (tp + fp + fn + 1e-7)
        sensitivity = tp / (tp + fn + 1e-7)
        specificity = tn / (tn + fp + 1e-7)

        return {
            "Dice": dice,
            "IoU": iou,
            "Sens": sensitivity,
            "Spec": specificity
        }
