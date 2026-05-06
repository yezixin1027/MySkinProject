import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordGatedAttention(nn.Module):
    """坐标门控注意力：结合位置感知与语义门控"""
    def __init__(self, in_channels, gating_channels, reduction=16):
        super(CoordGatedAttention, self).__init__()
        # 坐标扫描器
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.relu = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

        # 门控信号处理
        self.gate_conv = nn.Sequential(
            nn.Conv2d(gating_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x, g):
        n, c, h, w = x.size()
        # 1. 坐标维度感知
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = self.relu(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        a_coord = self.conv_h(x_h).sigmoid() * self.conv_w(x_w.permute(0, 1, 3, 2)).sigmoid()

        # 2. 语义门控引导
        g_up = F.interpolate(g, size=(h, w), mode='bilinear', align_corners=True)
        a_gate = self.gate_conv(g_up).sigmoid()

        # 3. 双重加权输出
        return x * a_coord * a_gate