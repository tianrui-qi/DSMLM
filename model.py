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
        
        in_feature  = config.dim_frame[0]  # input feature/channel/depth num
        up_c        = config.up_sample[0]  # upsampling scale, channel/depth
        up_hw       = config.up_sample[1]  # upsampling scale, high and wide

        self.input    = nn.ConvTranspose2d(
            in_feature       , up_c*in_feature  , up_hw, stride=up_hw)
        self.encoder1 = conv_block(
            in_feature*up_c*1, in_feature*up_c*2)
        self.pool     = nn.MaxPool2d(2)
        self.encoder2 = conv_block(
            in_feature*up_c*2, in_feature*up_c*4)
        self.upconv   = nn.ConvTranspose2d(
            in_feature*up_c*4, in_feature*up_c*2, 2, stride=2)
        self.decoder1 = conv_block(
            in_feature*up_c*4, in_feature*up_c*2)
        self.output   = nn.Sequential(
            nn.Conv2d(in_feature*up_c*2, in_feature*up_c, 1),
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
    def __init__(self, config):
        super().__init__()

        # for MSE loss between prediction and label
        self.mse    = nn.MSELoss()
        self.filter = torchvision.transforms.GaussianBlur(
            config.filter_size, sigma=config.filter_sigma)
        self.pad_size = (config.filter_size, config.filter_size,
                         config.filter_size, config.filter_size)
        
        # for L1 norm of prediction
        self.l1_coeff = config.l1_coeff
        
    def forward(self, predi, label):
        # MSE loss between prediction and label
        mse_loss = self.mse(
            self.filter(nn.functional.pad(label, self.pad_size)), 
            nn.functional.pad(predi, self.pad_size)
            #self.filter(nn.functional.pad(predi, self.pad_size))
            )
        
        # L1 norm of prediction
        l1_loss = self.l1_coeff * torch.norm(predi, p=1)
        
        return mse_loss + l1_loss
