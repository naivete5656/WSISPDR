import torch.nn as nn
from .network_parts import Up, DoubleConv
import torch.nn.functional as F
import torch


class UpIncBoundary(Up):
    def __init__(self, in_ch, inter_ch, out_ch):
        super().__init__(in_ch + inter_ch, int(in_ch + inter_ch / 2))
        self.up1 = nn.ConvTranspose2d(in_ch, inter_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch + inter_ch, out_ch)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = torch.cat([x, x3], dim=1)
        return x


class DoubleDilatedConv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Dilatedc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleDilatedConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class DilatedDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), DoubleDilatedConv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x

class MergeConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.conv(x)
        return x
