import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

try:
    from unet_parts import ResidualBlock, Down, OutConv
    from attention_gate import CoordGatedAttention
except ImportError:
    from .unet_parts import ResidualBlock, Down, OutConv
    from .attention_gate import CoordGatedAttention

class Up(nn.Module):
    """组装版上采样：放大 + 过滤 (Attention) + 拼接 (Concat)"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ResidualBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResidualBlock(in_channels, out_channels)

        # 实例化刚才写好的注意力模块
        self.att = CoordGatedAttention(in_channels // 2, in_channels // 2)

    def forward(self, x1, x2):
        x1_up = self.up(x1)
        # 重点：先用 x1 (深层) 对 x2 (浅层细节) 进行注意力过滤
        x2_att = self.att(x2, x1)

        # 补齐因奇数像素可能导致的尺寸差异
        diff_y = x2_att.size()[2] - x1_up.size()[2]
        diff_x = x2_att.size()[3] - x1_up.size()[3]
        x1_up = F.pad(x1_up, [diff_x // 2, diff_x - diff_x // 2,
                                diff_y // 2, diff_y - diff_y // 2])

        return self.conv(torch.cat([x2_att, x1_up], dim=1))

class ResCoordUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, bilinear=True, base_c=64):
        super(ResCoordUNet, self).__init__()
        self.in_conv = ResidualBlock(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 编码
        c1 = self.in_conv(x)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        c5 = self.down4(c4)
        # 解码
        d1 = self.up1(c5, c4)
        d2 = self.up2(d1, c3)
        d3 = self.up3(d2, c2)
        d4 = self.up4(d3, c1)
        return {"out": self.out_conv(d4)}

# 测试代码
if __name__ == "__main__":
    net = ResCoordUNet(in_channels=3, num_classes=1)
    img = torch.randn(1, 3, 224, 224)
    print("正在测试模型通路...")
    out = net(img)["out"]
    print(f"测试成功！输出形状: {out.shape}")