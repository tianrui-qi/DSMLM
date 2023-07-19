import torch
import torch.nn as nn
from torch import Tensor


class _DualConv(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super(_DualConv, self).__init__()
        if   dim == 2: Conv, BatchNorm = nn.Conv2d, nn.BatchNorm2d
        elif dim == 3: Conv, BatchNorm = nn.Conv3d, nn.BatchNorm3d
        else: raise ValueError("dim must be 2 or 3")

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


class _ResDualConv(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super(_ResDualConv, self).__init__()
        if   dim == 2: Conv, BatchNorm = nn.Conv2d, nn.BatchNorm2d
        elif dim == 3: Conv, BatchNorm = nn.Conv3d, nn.BatchNorm3d
        else: raise ValueError("dim must be 2 or 3")

        self.conv1 = nn.Sequential(
            Conv(in_channels, in_channels, 3, padding=1, bias=False),
            BatchNorm(in_channels)
        )
        self.conv2 = nn.Sequential(
            Conv(in_channels, out_channels, 3, padding=1, bias=False),
            BatchNorm(out_channels)
        )
        self.relu = nn.ReLU()

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


class _AttBlock(nn.Module):
    def __init__(self, in_channels, n_coefficients, dim):
        super(_AttBlock, self).__init__()
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


class ResAttUNet_L1(nn.Module):
    def __init__(self, config) -> None:
        super(ResAttUNet_L1, self).__init__()
        self.dim   = config.dim
        self.feats = config.feats
        self.residual  = config.residual
        self.attention = config.attention

        if self.residual: DualConv = _ResDualConv
        else:             DualConv = _DualConv

        if   self.dim == 2: MaxPool = nn.MaxPool2d
        elif self.dim == 3: MaxPool = nn.MaxPool3d
        else: raise ValueError("dim must be 2 or 3")

        self.encoder1 = DualConv(self.feats[0], self.feats[1], self.dim)
        self.maxpool1 = MaxPool(2)
        self.encoder2 = DualConv(self.feats[1], self.feats[2], self.dim)
        self.upconv1  = _UpConv(self.feats[2], self.feats[1], self.dim)
        if self.attention:
            self.att1 = _AttBlock(self.feats[1], self.feats[1]//2, self.dim)
        self.decoder1 = DualConv(self.feats[2], self.feats[1], self.dim)
        self.outconv  = _OutConv(self.feats[1], self.feats[0], self.dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.dim == 3: x = x.unsqueeze(1)

        enc1 = self.encoder1(x)
        x = self.maxpool1(enc1)
        x = self.encoder2(x)
        x = self.upconv1(x)
        if self.attention: enc1 = self.att1(enc1, x)
        x = torch.cat((enc1, x), dim=1)
        x = self.decoder1(x)
        x = self.outconv(x)

        if self.dim == 3: x = x.squeeze(1)
        return x


class ResAttUNet_L2(nn.Module):
    def __init__(self, config) -> None:
        super(ResAttUNet_L2, self).__init__()
        self.dim   = config.dim
        self.feats = config.feats
        self.residual  = config.residual
        self.attention = config.attention

        if self.residual: DualConv = _ResDualConv
        else:             DualConv = _DualConv

        if   self.dim == 2: MaxPool = nn.MaxPool2d
        elif self.dim == 3: MaxPool = nn.MaxPool3d
        else: raise ValueError("dim must be 2 or 3")

        self.encoder1 = DualConv(self.feats[0], self.feats[1], self.dim)
        self.maxpool1 = MaxPool(2)
        self.encoder2 = DualConv(self.feats[1], self.feats[2], self.dim)
        self.maxpool2 = MaxPool(2)
        self.encoder3 = DualConv(self.feats[2], self.feats[3], self.dim)
        self.upconv2  = _UpConv(self.feats[3], self.feats[2], self.dim)
        if self.attention: 
            self.att2 = _AttBlock(self.feats[2], self.feats[2]//2, self.dim)
        self.decoder2 = DualConv(self.feats[3], self.feats[2], self.dim)
        self.upconv1  = _UpConv(self.feats[2], self.feats[1], self.dim)
        if self.attention: 
            self.att1 = _AttBlock(self.feats[1], self.feats[1]//2, self.dim)
        self.decoder1 = DualConv(self.feats[2], self.feats[1], self.dim)
        self.outconv  = _OutConv(self.feats[1], self.feats[0], self.dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.dim == 3: x = x.unsqueeze(1)

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
        x = self.outconv(x)

        if self.dim == 3: x = x.squeeze(1)
        return x
