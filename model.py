import torch
from torch import nn
import torch.nn.functional as F


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

        self.kernel_size  = config.kernel_size
        self.kernel_sigma = config.kernel_sigma

        self.kernel = self.gaussian_kernel(
            3, self.kernel_size, self.kernel_sigma)
        self.kernel = self.kernel.reshape(
            1, 1, *self.kernel.shape)  # [D H W] to  [C_out C_in D H W]
        self.pad = [self.kernel_size for _ in range(6)]  # [C H W]

    def forward(self, predi, label):
        return F.mse_loss(
            self.gaussian_blur_3d(F.pad(predi, self.pad)), 
            self.gaussian_blur_3d(F.pad(label, self.pad)), 
            reduction='none').sum()

    # help function for convolve a frame with a Gaussian

    def gaussian_kernel(self, dim, kernel_size, kernel_sigma):
        # build 1D Gaussian kernel
        r = kernel_size // 2
        coord = torch.linspace(-r, r, kernel_size)
        kernel_1d  = torch.exp((-(coord / kernel_sigma)**2 / 2))
        kernel_1d /= kernel_1d.sum()  # Normalization

        # build nd Gaussian kernel using einsum
        equation = ','.join(f'{chr(97 + i)}' for i in range(dim))
        operands = [kernel_1d for _ in range(dim)]
        kernel_nd  = torch.einsum(equation, *operands)
        kernel_nd /= kernel_nd.sum()  # Normalization

        return kernel_nd 

    def gaussian_blur_3d(self, frame):
        return F.conv3d(
            frame.unsqueeze(1),  # [B C H W] to [B C D H W]
            self.kernel,  # [C_out C_in D H W]
            stride=1, padding=self.kernel_size//2).reshape(*frame.shape)
