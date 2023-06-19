import torch
from torch import nn
import torch.nn.functional as F


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
        #up   = self.input(x)
        enc1 = self.encoder1(up)
        enc2 = self.pool(enc1)
        enc2 = self.encoder2(enc2)
        enc2 = self.upconv(enc2)
        dec2 = torch.cat((enc1, enc2), dim=1)
        dec2 = self.decoder1(dec2)
        return self.output(dec2)


class Criterion(nn.Module):
    def __init__(self, config):
        super().__init__()
        # configuration
        self.kernel_size  = config.kernel_size
        self.kernel_sigma = config.kernel_sigma
        self.l1_coeff     = config.l1_coeff

        # Gaussian kernel using help function gaussian_kernel
        self.kernel = self.gaussian_kernel(
            3, self.kernel_size, self.kernel_sigma)
        self.kernel = self.kernel.reshape(
            1, 1, *self.kernel.shape)  # [D H W] to  [C_out C_in D H W]
        
        # pad size, pad before convolve Gaussian kernel
        self.pad = [self.kernel_size for _ in range(6)]  # [C H W]

    def forward(self, predi, label):
        mse_loss = F.mse_loss(
            self.gaussian_blur_3d(F.pad(predi, self.pad)), 
            self.gaussian_blur_3d(F.pad(label, self.pad)), 
            reduction="sum") 
        l1_loss = F.l1_loss(predi, torch.zeros_like(predi), reduction="sum")
        return (mse_loss + self.l1_coeff * l1_loss) / len(predi)

    def to(self, device):
        # Call the original 'to' method to move parameters and buffers
        super(Criterion, self).to(device)
        self.kernel = self.kernel.to(device)
        return self

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
