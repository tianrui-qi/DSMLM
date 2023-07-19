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


class ResAttUNet(nn.Module):
    def __init__(self, config) -> None:
        super(ResAttUNet, self).__init__()
        self.dim   = config.dim
        self.feats = config.feats
        self.residual  = config.residual
        self.attention = config.attention

        if self.residual: DualConv = _ResDualConv
        else:             DualConv = _DualConv

        if   self.dim == 2: MaxPool = nn.MaxPool2d
        elif self.dim == 3: MaxPool = nn.MaxPool3d
        else: raise ValueError("dim must be 2 or 3")

        # encoder
        self.encoder = nn.ModuleList([
            DualConv(self.feats[i], self.feats[i+1], self.dim) 
            for i in range(len(self.feats)-1)
        ])
        self.maxpool = nn.ModuleList([
            MaxPool(2) 
            for _ in range(len(self.feats)-2)
        ])
        # decoder
        self.upconv = nn.ModuleList([
            _UpConv(self.feats[i+1], self.feats[i], self.dim)
            for i in range(1, len(self.feats)-1)
        ])
        if self.attention: self.att = nn.ModuleList([
            _AttBlock(self.feats[i], self.feats[i]//2, self.dim)
            for i in range(1, len(self.feats)-1)
        ])
        self.decoder = nn.ModuleList([
            DualConv(self.feats[i+1], self.feats[i], self.dim)
            for i in range(1, len(self.feats)-1)
        ])
        # output
        self.outconv = _OutConv(self.feats[1], self.feats[0], self.dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.dim == 3: x = x.unsqueeze(1)
        # encoder
        enc = [self.encoder[0](x)]
        for i in range(1, len(self.feats)-1):
            if i != len(self.feats)-2: 
                enc.append(self.encoder[i](self.maxpool[i-1](enc[i-1])))
            else:
                x = self.encoder[i](self.maxpool[i-1](enc[i-1]))
        # decoder
        for i in range(1, len(self.feats)-1):
            x = self.upconv[-i](x)
            if self.attention: enc[-i] = self.att[-i](enc[-i], x)
            x = torch.cat((enc[-i], x), dim=1)
            x = self.decoder[-i](x)
        # output
        x = self.outconv(x)

        if self.dim == 3: x = x.squeeze(1)
        return x
