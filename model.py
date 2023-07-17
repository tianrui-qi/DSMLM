import torch
import torch.nn as nn
from torch import Tensor


__all__ = [
    "ResUNet2D", "ResUNet3D",
    "getModel"
]


class _UNetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(_UNetBlock2D, self).__init__()
        # UNet Component
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out  = self.conv1(x)
        out  = self.relu(out)
        out  = self.conv2(out)
        out  = self.relu(out)
        return out


class _UNetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(_UNetBlock3D, self).__init__()
        # UNet Component
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out  = self.conv1(x)
        out  = self.relu(out)
        out  = self.conv2(out)
        out  = self.relu(out)
        return out


class _ResUNetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_ResUNetBlock2D, self).__init__()
        # UNet Component
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()
        # Residual Component
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels == out_channels: self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out  = self.conv1(x)
        out  = self.relu(out)
        out  = self.conv2(out)
        out += self.skip(residual)
        out  = self.relu(out)
        return out


class _ResUNetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_ResUNetBlock3D, self).__init__()
        # UNet Component
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        self.relu = nn.ReLU()
        # Residual Component
        self.skip = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        if in_channels == out_channels: self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out  = self.conv1(x)
        out  = self.relu(out)
        out  = self.conv2(out)
        out += self.skip(residual)
        out  = self.relu(out)
        return out


class ResUNet2D(nn.Module):
    def __init__(self, config) -> None:
        super(ResUNet2D, self).__init__()
        base = config.dim_frame[0] * config.up_sample[0]
        self.input    = nn.Upsample(scale_factor=tuple(config.up_sample))
        self.encoder1 = _ResUNetBlock2D(base*1, base*2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.encoder2 = _ResUNetBlock2D(base*2, base*4)
        self.maxpool2 = nn.MaxPool2d(2)
        self.encoder3 = _ResUNetBlock2D(base*4, base*8)
        self.up_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(base*8, base*4, 3, padding=1)
        )
        self.decoder2 = _ResUNetBlock2D(base*8, base*4)
        self.up_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(base*4, base*2, 3, padding=1)
        )
        self.decoder1 = _ResUNetBlock2D(base*4, base*2)
        self.output_c = nn.Conv2d(base*2, base, 1)
        self.output_f = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.input(x.unsqueeze(1)).squeeze(1)
        enc1 = self.encoder1(x)
        enc2 = self.maxpool1(enc1)
        enc2 = self.encoder2(enc2)
        out = self.maxpool2(enc2)
        out = self.encoder3(out)
        out = self.up_conv2(out)
        out = torch.cat((enc2, out), dim=1)
        out = self.decoder2(out)
        out = self.up_conv1(out)
        out = torch.cat((enc1, out), dim=1)
        out = self.decoder1(out)
        return self.output_f(self.output_c(out)+x)


class ResUNet3D(nn.Module):
    def __init__(self, config) -> None:
        super(ResUNet3D, self).__init__()
        base = config.base
        self.intput   = nn.Upsample(scale_factor=tuple(config.up_sample))
        self.encoder1 = _ResUNetBlock3D(1, base)
        self.maxpool1 = nn.MaxPool3d(2)
        self.encoder2 = _ResUNetBlock3D(base*1, base*2)
        self.maxpool2 = nn.MaxPool3d(2)
        self.encoder3 = _ResUNetBlock3D(base*2, base*4)
        self.up_conv2  = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(base*4, base*2, kernel_size=3, padding=1)
        )
        self.decoder2 = _ResUNetBlock3D(base*4, base*2)
        self.up_conv1  = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(base*2, base*1, kernel_size=3, padding=1)
        )
        self.decoder1 = _ResUNetBlock3D(base*2, base*1)
        self.output_c = nn.Conv3d(base*1, 1, kernel_size=1)
        self.output_f = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.intput(x.unsqueeze(1))
        enc1 = self.encoder1(x)
        enc2 = self.maxpool1(enc1)
        enc2 = self.encoder2(enc2)
        out = self.maxpool2(enc2)
        out = self.encoder3(out)
        out = self.up_conv2(out)
        out = torch.cat((enc2, out), dim=1)
        out = self.decoder2(out)
        out = self.up_conv1(out)
        out = torch.cat((enc1, out), dim=1)
        out = self.decoder1(out)
        return self.output_f(self.output_c(out)+x).squeeze(1)


def getModel(config):
    if config.type_model == "ResUNet2D":
        return ResUNet2D(config)
    elif config.type_model == "ResUNet3D":
        return ResUNet3D(config)
    else:
        raise ValueError("Only ResUNet2D and ResUNet3D are supported.")
