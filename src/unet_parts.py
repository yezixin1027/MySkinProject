import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """带残差连接的双卷积块：[卷积+BN+ReLU] * 2 + Shortcut"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(ResidualBlock, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels

        # 主路径
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # 快捷路径 (转接头)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 核心：加工后的特征 + 原始特征
        return self.relu(self.conv_path(x) + self.shortcut(x))


class Down(nn.Sequential):
    """下采样：缩小尺寸 (MaxPool) + 提取特征 (ResidualBlock)"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            ResidualBlock(in_channels, out_channels)
        )


class OutConv(nn.Module):
    """输出层：利用 1x1 卷积将特征图压缩到指定类别数"""

    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)