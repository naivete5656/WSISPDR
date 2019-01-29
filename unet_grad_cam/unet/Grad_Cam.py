import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import *
from .unet_model import UNet


class GradCamUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(GradCamUNet, self).__init__()
        self.UNet = UNet(n_channels, n_classes)
        self.uconv1 = self.UNet.up2.conv.conv._modules['0']
        self.ubn1 = self.UNet.up2.conv.conv._modules['1']
        self.ure1 = self.UNet.up2.conv.conv._modules['2']
        self.uconv2 = self.UNet.up2.conv.conv._modules['3']
        self.ubn2 = self.UNet.up2.conv.conv._modules['4']
        self.ure2 = self.UNet.up2.conv.conv._modules['5']
        self.oconv = self.UNet.outc.conv
        self.sigmoid = nn.Sigmoid()
        self.gradients = []
        self.mp = self.UNet.down4.mpconv._modules['0']
        self.conv3 = self.UNet.down4.mpconv._modules['1'].conv._modules['0']
        self.bn3 = self.UNet.down4.mpconv._modules['1'].conv._modules['1']
        self.re3 = self.UNet.down4.mpconv._modules['1'].conv._modules['2']
        self.conv4 = self.UNet.down4.mpconv._modules['1'].conv._modules['3']
        self.bn4 = self.UNet.down4.mpconv._modules['1'].conv._modules['4']
        self.re4 = self.UNet.down4.mpconv._modules['1'].conv._modules['5']
        self.iconv = self.UNet.inc.conv.conv._modules['0']
        self.ibn = self.UNet.inc.conv.conv._modules['1']
        self.ire = self.UNet.inc.conv.conv._modules['2']
        self.iconv2 = self.UNet.inc.conv.conv._modules['3']
        self.ibn2 = self.UNet.inc.conv.conv._modules['4']
        self.ire2 = self.UNet.inc.conv.conv._modules['5']

    def forward(self, x):
        x = self.UNet(x)
        return x

    def for_cam(self, x):
        x1 = self.UNet.inc(x)
        x2 = self.UNet.down1(x1)
        x3 = self.UNet.down2(x2)
        x4 = self.UNet.down3(x3)
        x5 = self.UNet.down4(x4)
        x = self.UNet.up1(x5, x4)

        y = x3
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        diffx = x.size()[2] - y.size()[2]
        diffy = x.size()[3] - y.size()[3]
        y = F.pad(y, (diffx // 2, int(diffx / 2),
                        diffy // 2, int(diffy / 2)))
        x = torch.cat([y, x], dim=1)
        x_feature = self.uconv1(x)
        x = self.ubn1(x_feature)
        x = self.ure1(x)
        x = self.uconv2(x)
        x = self.ubn2(x)
        x = self.ure2(x)

        # x = self.UNet.up2(x, x3)
        x = self.UNet.up3(x, x2)
        x = self.UNet.up4(x, x1)
        x = self.UNet.outc(x)
        return x, x_feature

    def for_cam2(self, x):
        x1 = self.UNet.inc(x)
        x2 = self.UNet.down1(x1)
        x3 = self.UNet.down2(x2)
        x4 = self.UNet.down3(x3)
        x5 = self.UNet.down4(x4)
        x = self.UNet.up1(x5, x4)
        x = self.UNet.up2(x, x3)
        x = self.UNet.up3(x, x2)
        x = self.UNet.up4(x, x1)
        x_feature = self.oconv(x)
        x = self.sigmoid(x_feature)
        return x, x_feature

    def for_cam3(self, x):
        x1 = self.UNet.inc(x)
        x2 = self.UNet.down1(x1)
        x3 = self.UNet.down2(x2)
        x4 = self.UNet.down3(x3)
        x = self.mp(x4)
        x_feature = self.conv3(x)
        x = self.bn3(x_feature)
        x = self.re3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x5 = self.re4(x)
        x = self.UNet.up1(x5, x4)
        x = self.UNet.up2(x, x3)
        x = self.UNet.up3(x, x2)
        x = self.UNet.up4(x, x1)
        x = self.UNet.outc(x)
        return x, x_feature

    def for_cam4(self, x):
        x = self.iconv(x)
        x = self.ibn(x)
        x = self.ire(x)
        x_feature = self.iconv2(x)
        x = self.ibn2(x_feature)
        x1 = self.ire2(x)
        x2 = self.UNet.down1(x1)
        x3 = self.UNet.down2(x2)
        x4 = self.UNet.down3(x3)
        x5 = self.UNet.down4(x4)
        x = self.UNet.up1(x5, x4)
        x = self.UNet.up2(x, x3)
        x = self.UNet.up3(x, x2)
        x = self.UNet.up4(x, x1)
        x = self.UNet.outc(x)
        return x, x_feature

    def save_gradient(self, grad):
        self.gradients.append(grad)



