# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
from .network_parts import DoubleConv


class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownConv, self).__init__()
        self.mpconv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch,kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(in_ch),
            DoubleConv(in_ch, out_ch)

        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class DownInternal(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DownInternal, self).__init__()
        self.mp = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, x1):
        x = self.mp(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv(x)
        return x

class Outconv9(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv9, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x
