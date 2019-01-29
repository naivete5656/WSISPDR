import torch.nn as nn
from .network_parts import Up, DoubleConv
import torch.nn.functional as F
import torch


class UpIncBoundary(Up):
    def __init__(self, in_ch, inter_ch, out_ch):
        super().__init__(in_ch + inter_ch, out_ch)
        self.up1 = nn.ConvTranspose2d(in_ch, inter_ch, kernel_size=2, stride=2)

    def forward(self, x1, x2, x3):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = torch.cat([x, x3], dim=1)
        return x
