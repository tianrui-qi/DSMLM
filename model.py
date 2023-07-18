import torch
import torch.nn as nn
from torch import Tensor


class _UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super(_UNetBlock, self).__init__()
        if   dim == 2: Conv, BatchNorm = nn.Conv2d, nn.BatchNorm2d
        elif dim == 3: Conv, BatchNorm = nn.Conv3d, nn.BatchNorm3d
        else: raise ValueError("dim must be 2 or 3")

        # UNet Component
        self.conv1 = nn.Sequential(
            Conv(in_channels, in_channels, 3, padding=1, bias=False),
            BatchNorm(in_channels)
        )
        self.conv2 = nn.Sequential(
            Conv(in_channels, out_channels, 3, padding=1, bias=False),
            BatchNorm(out_channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x  = self.conv1(x)
        x  = self.relu(x)
        x  = self.conv2(x)
        x  = self.relu(x)
        return x


class _ResUNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super(_ResUNetBlock, self).__init__()
        if   dim == 2: Conv, BatchNorm = nn.Conv2d, nn.BatchNorm2d
        elif dim == 3: Conv, BatchNorm = nn.Conv3d, nn.BatchNorm3d
        else: raise ValueError("dim must be 2 or 3")

        # UNet Component
        self.conv1 = nn.Sequential(
            Conv(in_channels, in_channels, 3, padding=1, bias=False),
            BatchNorm(in_channels)
        )
        self.conv2 = nn.Sequential(
            Conv(in_channels, out_channels, 3, padding=1, bias=False),
            BatchNorm(out_channels)
        )
        self.relu = nn.ReLU()
        # Residual Component
        self.skip = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=1, bias=False),
            BatchNorm(out_channels),
        )
        if in_channels == out_channels: self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x  = self.conv1(x)
        x  = self.relu(x)
        x  = self.conv2(x)
        x += self.skip(residual)
        x  = self.relu(x)
        return x


class _UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super(_UpConv, self).__init__()
        if   dim == 2: Conv, BatchNorm = nn.Conv2d, nn.BatchNorm2d
        elif dim == 3: Conv, BatchNorm = nn.Conv3d, nn.BatchNorm3d
        else: raise ValueError("dim must be 2 or 3")

        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(in_channels, out_channels, 3, padding=1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.upconv(x)


class _AttentionBlock(nn.Module):
    def __init__(self, in_channels, n_coefficients, dim):
        super(_AttentionBlock, self).__init__()
        if   dim == 2: Conv, BatchNorm = nn.Conv2d, nn.BatchNorm2d
        elif dim == 3: Conv, BatchNorm = nn.Conv3d, nn.BatchNorm3d
        else: raise ValueError("dim must be 2 or 3")

        self.w_decod = nn.Sequential(
            Conv(in_channels, n_coefficients, 1, bias=False),
            BatchNorm(n_coefficients)
        )
        self.w_encod = nn.Sequential(
            Conv(in_channels, n_coefficients, 1, bias=False),
            BatchNorm(n_coefficients)
        )
        self.relu = nn.ReLU()
        self.psi = nn.Sequential(
            Conv(n_coefficients, 1, 1, bias=False),
            BatchNorm(1),
            nn.Sigmoid()
        )

    def forward(self, encod, decod):
        psi = self.w_encod(encod) + self.w_decod(decod)
        psi = self.relu(psi)
        psi = self.psi(psi)
        return encod * psi


class _OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super(_OutConv, self).__init__()
        if   dim == 2: Conv = nn.Conv2d
        elif dim == 3: Conv = nn.Conv3d
        else: raise ValueError("dim must be 2 or 3")

        self.outconv   = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.outconv(x)


class ResAttUNet_2DL1(nn.Module):
    def __init__(self, config) -> None:
        super(ResAttUNet_2DL1, self).__init__()
        base = config.base
        self.residual  = config.residual
        self.attention = config.attention

        if self.residual: CoderBlock = _ResUNetBlock
        else:             CoderBlock = _UNetBlock

        dim  = 2
        self.encoder1 = CoderBlock(base*1, base*2, dim)
        self.maxpool1 = nn.MaxPool2d(2)
        self.encoder2 = CoderBlock(base*2, base*4, dim)
        self.upconv1  = _UpConv(base*4, base*2, dim)
        if self.attention: self.att1 = _AttentionBlock(base*2, base, dim)
        self.decoder1 = CoderBlock(base*4, base*2, dim)
        self.outconv  = _OutConv(base*2, base, dim)

    def forward(self, x: Tensor) -> Tensor:
        enc1 = self.encoder1(x)
        x = self.maxpool1(enc1)
        x = self.encoder2(x)
        x = self.upconv1(x)
        if self.attention: enc1 = self.att1(enc1, x)
        x = torch.cat((enc1, x), dim=1)
        x = self.decoder1(x)
        return self.outconv(x)


class ResAttUNet_2DL2(nn.Module):
    def __init__(self, config) -> None:
        super(ResAttUNet_2DL2, self).__init__()
        base = config.base
        self.residual  = config.residual
        self.attention = config.attention

        if self.residual: CoderBlock = _ResUNetBlock
        else:             CoderBlock = _UNetBlock

        dim  = 2
        self.encoder1 = CoderBlock(base*1, base*2, dim)
        self.maxpool1 = nn.MaxPool2d(2)
        self.encoder2 = CoderBlock(base*2, base*4, dim)
        self.maxpool2 = nn.MaxPool2d(2)
        self.encoder3 = CoderBlock(base*4, base*8, dim)
        self.upconv2  = _UpConv(base*8, base*4, dim)
        if self.attention: self.att2 = _AttentionBlock(base*4, base*2, dim)
        self.decoder2 = CoderBlock(base*8, base*4, dim)
        self.upconv1  = _UpConv(base*4, base*2, dim)
        if self.attention: self.att1 = _AttentionBlock(base*2, base*1, dim)
        self.decoder1 = CoderBlock(base*4, base*2, dim)
        self.outconv  = _OutConv(base*2, base, dim)

    def forward(self, x: Tensor) -> Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.maxpool1(enc1)
        enc2 = self.encoder2(enc2)
        x = self.maxpool2(enc2)
        x = self.encoder3(x)
        x = self.upconv2(x)
        if self.attention: enc2 = self.att2(enc2, x)
        x = torch.cat((enc2, x), dim=1)
        x = self.decoder2(x)
        x = self.upconv1(x)
        if self.attention: enc1 = self.att1(enc1, x)
        x = torch.cat((enc1, x), dim=1)
        x = self.decoder1(x)
        return self.outconv(x)


class ResUNet_3DL1(nn.Module):
    def __init__(self, config) -> None:
        super(ResUNet_3DL1, self).__init__()
        base = config.base
        dim  = 3
        self.encoder1 = _ResUNetBlock(1, base, dim)
        self.maxpool1 = nn.MaxPool3d(2)
        self.encoder2 = _ResUNetBlock(base*1, base*2, dim)
        self.upconv1  = _UpConv(base*2, base*1, dim)
        self.decoder1 = _ResUNetBlock(base*2, base*1, dim)
        self.outconv  = _OutConv(base*1, 1, dim)

    def forward(self, x: Tensor) -> Tensor:
        enc1 = self.encoder1(x.unsqueeze(1))
        x = self.maxpool1(enc1)
        x = self.encoder2(x)
        x = self.upconv1(x)
        x = torch.cat((enc1, x), dim=1)
        x = self.decoder1(x)
        return self.outconv(x).squeeze(1)


class ResUNet_3DL2(nn.Module):
    def __init__(self, config) -> None:
        super(ResUNet_3DL2, self).__init__()
        base = config.base
        dim = 3
        self.encoder1 = _ResUNetBlock(1, base, dim)
        self.maxpool1 = nn.MaxPool3d(2)
        self.encoder2 = _ResUNetBlock(base*1, base*2, dim)
        self.maxpool2 = nn.MaxPool3d(2)
        self.encoder3 = _ResUNetBlock(base*2, base*4, dim)
        self.upconv2  = _UpConv(base*4, base*2, dim)
        self.decoder2 = _ResUNetBlock(base*4, base*2, dim)
        self.upconv1  = _UpConv(base*2, base*1, dim)
        self.decoder1 = _ResUNetBlock(base*2, base*1, dim)
        self.outconv  = _OutConv(base*1, 1, dim)

    def forward(self, x: Tensor) -> Tensor:
        enc1 = self.encoder1(x.unsqueeze(1))
        enc2 = self.maxpool1(enc1)
        enc2 = self.encoder2(enc2)
        x = self.maxpool2(enc2)
        x = self.encoder3(x)
        x = self.upconv2(x)
        x = torch.cat((enc2, x), dim=1)
        x = self.decoder2(x)
        x = self.upconv1(x)
        x = torch.cat((enc1, x), dim=1)
        x = self.decoder1(x)
        return self.outconv(x).squeeze(1)
