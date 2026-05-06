import torch
import numpy as np


class SegmentationMetric:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def calculate_all(self, predict, target):
        """
        计算所有关键指标
        predict: 模型的输出 (Tensor, 通常为 sigmoid 后的结果或 logits)
        target: 真实的标签 (Tensor, mask)
        """
        # 如果输入是 logits，先进行 sigmoid 处理
        if predict.max() > 1.0 or predict.min() < 0.0:
            predict = torch.sigmoid(predict)

        # 根据阈值转换为二值掩码 (0/1)
        predict = (predict > self.threshold).float()
        target = (target > 0.5).float()

        # 展平 Tensor 并转为 numpy 进行逻辑运算
        predict = predict.view(-1).cpu().numpy()
        target = target.view(-1).cpu().numpy()

        # 计算基础统计量
        tp = np.sum((predict == 1) & (target == 1))  # 真阳性
        fp = np.sum((predict == 1) & (target == 0))  # 假阳性
        tn = np.sum((predict == 0) & (target == 0))  # 真阴性
        fn = np.sum((predict == 0) & (target == 1))  # 假阴性

        # 核心指标计算
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
        iou = tp / (tp + fp + fn + 1e-7)

        # Sensitivity (真阳性率) - 捕捉病灶的能力
        sensitivity = tp / (tp + fn + 1e-7)

        # Specificity (真阴性率) - 排除健康区域的能力
        specificity = tn / (tn + fp + 1e-7)

        return {
            "Dice": dice,
            "IoU": iou,
            "Sens": sensitivity,
            "Spec": specificity
        }