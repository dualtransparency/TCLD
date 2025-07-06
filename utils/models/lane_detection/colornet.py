# import ptvsd
# ptvsd.enable_attach(address=('10.0.0.2', 1234))
# ptvsd.wait_for_attach()

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from ..builder import MODELS


@MODELS.register()
class ColorNet(nn.Module):

    def __init__(self,
                 backbone_cfg,):

        super().__init__()

        self.extractor = MODELS.from_dict(backbone_cfg)
        self.decoder = ColorDecoder()

    def forward(self, x):
        features = self.extractor(x)

        output = self.decoder(features)

        return output + x, output


class ColorDecoder(nn.Module):

    def __init__(self, input_channels=256, up_sample_ratio=16):
        super().__init__()

        self.conv1 = single_conv(256, 256)
        self.up1 = up(256)

        self.conv2 = single_conv(128, 128)
        self.up2 = up(128)

        self.conv3 = single_conv(64, 64)
        self.up3 = up(64)

        self.conv4 = single_conv(32, 32)
        self.up4 = up(32)

        self.out = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
        )
    
    def forward(self, input):

        x = self.conv1(input['layer3'])
        x = self.up1(x, input['layer2'])

        x = self.conv2(x)
        x = self.up2(x, input['layer1'])

        x = self.conv3(x)
        x = self.up3(x)

        x = self.conv4(x)
        x = self.up4(x)

        x = self.out(x)

        return x

        
class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        
        if x2 is None:
            return x1
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = x2 + x1
        return x

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.conv(x)