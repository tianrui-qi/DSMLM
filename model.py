import torch
from torch import nn
import torchvision


class UNet2D(nn.Module):
    def __init__(self, config):
        super(UNet2D, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        
        in_feature    = config.dim_frame[0]
        self.input    = nn.ConvTranspose2d(in_feature, 4*in_feature, 8, stride=8)
        self.encoder1 = conv_block(4*in_feature, 8*in_feature)
        self.pool     = nn.MaxPool2d(2)
        self.encoder2 = conv_block(8*in_feature, 16*in_feature)
        self.upconv   = nn.ConvTranspose2d(16*in_feature, 8*in_feature, 2, stride=2)
        self.decoder1 = conv_block(16*in_feature, 8*in_feature)
        self.output   = nn.Sequential(
            nn.Conv2d(8*in_feature, 4*in_feature, 1),
            nn.ReLU()
        )
        
    def forward(self, x):
        up   = self.input(x)
        enc1 = self.encoder1(up)
        enc2 = self.pool(enc1)
        enc2 = self.encoder2(enc2)
        enc2 = self.upconv(enc2)
        dec2 = torch.cat((enc1, enc2), dim=1)
        dec2 = self.decoder1(dec2)
        return self.output(dec2)


class DeepSTORMLoss(nn.Module):
    """
    v1: mse(output, label) + l1 norm
    v2: mse(output * G, label * G) + l1 norm
    v3: mse(output * G, label * G)
    """
    def __init__(self, config):
        super().__init__()
        self.gauss = config.gauss
        self.l1    = config.l1

        self.mse = nn.MSELoss()
        self.filter = torchvision.transforms.GaussianBlur(
            config.filter_size, sigma=config.filter_sigma)
        
    def forward(self, frame, label):
        if self.gauss:
            mse_loss = self.mse(self.filter(frame), self.filter(label))
        else:
            mse_loss = self.mse(frame, label)

        if self.l1:
            return mse_loss + 0.01 * torch.norm(frame, p=1)
        else:
            return mse_loss
