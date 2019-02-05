# full assembly of the sub-parts to form the complete net


from .unet_multi import *
from .unet_parts import *
import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = Inconv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = Outconv(64, n_classes)

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

    def output_internal_layer(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = F.max_pool2d(x4, 2)
        return x

    def outlast(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x


class UNetOnlyConv(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetOnlyConv, self).__init__()
        self.inc = Inconv(n_channels, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)
        self.down3 = DownConv(256, 512)
        self.down4 = DownConv(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = Outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class UNetInternalCascade(UNet):
    def __init__(self, n_channels, n_classes):
        super(UNetInternalCascade, self).__init__(n_channels, n_classes)
        self.down4 = DownInternal(1024, 512)

    def forward(self, x, y):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4, y)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

    def output_internal_layer(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = F.max_pool2d(x4, 2)
        return x


class UNetInternalCascadeContinue(UNet):
    def __init__(self, n_channels, n_classes):
        super(UNetInternalCascadeContinue, self).__init__(n_channels, n_classes)
        self.inc2 = Inconv(n_channels, 64)
        self.down5 = Down(64, 128)
        self.down6 = Down(128, 256)
        self.down7 = Down(256, 512)
        self.down8 = DownInternal(512, 512)
        self.up5 = Up(1024, 256)
        self.up6 = Up(512, 128)
        self.up7 = Up(256, 64)
        self.up8 = Up(128, 64)
        self.outc2 = Outconv(64, n_classes)

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        y1 = self.inc2(input)
        y2 = self.down5(y1)
        y3 = self.down6(y2)
        y4 = self.down7(y3)
        y5 = self.down8(y4, F.max_pool2d(x4, kernel_size=2))
        y = self.up5(y5, y4)
        y = self.up6(y, y3)
        y = self.up7(y, y2)
        y = self.up8(y, y1)
        y = self.outc2(y)
        return x


class UNetCascadeContinue(UNet):
    def __init__(self, n_channels, n_classes):
        super(UNetCascadeContinue, self).__init__(n_channels, n_classes)
        self.inc_9 = Inconv(n_channels, 64)
        self.down1_9 = Down(64, 128)
        self.down2_9 = Down(128, 256)
        self.down3_9 = Down(256, 512)
        self.down4_9 = Down(512, 512)
        self.up1_9 = Up(1024, 256)
        self.up2_9 = Up(512, 128)
        self.up3_9 = Up(256, 64)
        self.up4_9 = Up(128, 64)
        self.outc_9 = Outconv9(128, n_classes)

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x_before = self.up4(x, x1)
        x = self.outc(x_before)

        x1_9 = self.inc_9(input)
        x2_9 = self.down1_9(x1_9)
        x3_9 = self.down2_9(x2_9)
        x4_9 = self.down3_9(x3_9)
        x5_9 = self.down4_9(x4_9)
        x_9 = self.up1_9(x5_9, x4_9)
        x_9 = self.up2_9(x_9, x3_9)
        x_9 = self.up3_9(x_9, x2_9)
        x_9 = self.up4_9(x_9, x1_9)
        x_9 = self.outc_9(x_9, x_before)
        return x, x_9


class UNetCascade(UNet):
    def __init__(self, n_channels, n_classes):
        super(UNetCascade, self).__init__(n_channels, n_classes)
        self.inc_9 = Inconv(n_channels, 64)
        self.down1_9 = Down(64, 128)
        self.down2_9 = Down(128, 256)
        self.down3_9 = Down(256, 512)
        self.down4_9 = Down(512, 512)
        self.up1_9 = Up(1024, 256)
        self.up2_9 = Up(512, 128)
        self.up3_9 = Up(256, 64)
        self.up4_9 = Up(128, 64)
        self.outc_9 = Outconv9(128, n_classes)

    def forward(self, input1, input2):
        x1 = self.inc(input1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc_9(x, input2)
        return x


class UnetMultiFixedWeight(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UnetMultiFixedWeight, self).__init__()
        self.ex1 = Expert(n_channels, n_classes)
        self.ex2 = Expert(n_channels, n_classes)
        self.ex3 = Expert(n_channels, n_classes)
        self.finc = FinalConv(192, n_classes)
        self.x_w = nn.Parameter(torch.randn(1, requires_grad=True))
        self.y_w = nn.Parameter(torch.randn(1, requires_grad=True))
        self.z_w = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, input):
        x, x_f = self.ex1(input)
        y, y_f = self.ex2(input)
        z, z_f = self.ex3(input)
        res = self.finc(x_f * self.x_w, y_f * self.y_w, z_f * self.z_w)
        return x, y, z, res


class UnetMulti(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UnetMulti, self).__init__()
        self.ex1 = Expert(n_channels, n_classes)
        self.ex2 = Expert(n_channels, n_classes)
        self.ex3 = Expert(n_channels, n_classes)
        self.finc = FinalConv(192, n_classes)

    def forward(self, input):
        x, x_f = self.ex1(input)
        y, y_f = self.ex2(input)
        z, z_f = self.ex3(input)
        res = self.finc(x_f, y_f, z_f)
        return x, y, z, res
