import torch
from torch import nn


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    def __init__(self, config):
        super(UNet2D, self).__init__()
        
        in_feature  = config.dim_frame[0]  # input feature/channel/depth num
        up_c        = config.up_sample[0]  # upsampling scale, channel/depth
        up_hw       = config.up_sample[1]  # upsampling scale, high and wide

        self.input    = nn.Upsample(
            scale_factor=tuple(config.up_sample), mode='nearest')
        #self.input    = nn.ConvTranspose2d(
        #    in_feature, up_c*in_feature, up_hw, stride=up_hw)

        self.encoder1 = UNetBlock(in_feature*up_c*1, in_feature*up_c*2)
        self.pool     = nn.MaxPool2d(2)
        self.encoder2 = UNetBlock(in_feature*up_c*2, in_feature*up_c*4)
        self.upconv   = nn.ConvTranspose2d(
            in_feature*up_c*4, in_feature*up_c*2, 4, stride=2, padding=1)
        self.decoder1 = UNetBlock(in_feature*up_c*4, in_feature*up_c*2)

        self.output   = nn.Sequential(
            nn.Conv2d(in_feature*up_c*2, in_feature*up_c, 1),
            nn.ReLU())

    def forward(self, x):
        up   = self.input(x.unsqueeze(1)).squeeze(1)
        enc1 = self.encoder1(up)
        enc2 = self.pool(enc1)
        enc2 = self.encoder2(enc2)
        enc2 = self.upconv(enc2)
        dec2 = torch.cat((enc1, enc2), dim=1)
        dec2 = self.decoder1(dec2)
        return self.output(dec2)
