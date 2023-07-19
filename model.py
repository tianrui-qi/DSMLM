import torch
import torch.nn as nn
from torch import Tensor


class _DualConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dim: int) -> None:
        super(_DualConv, self).__init__()
        if   dim == 2: Conv, BatchNorm = nn.Conv2d, nn.BatchNorm2d
        elif dim == 3: Conv, BatchNorm = nn.Conv3d, nn.BatchNorm3d
        else: raise ValueError("dim must be 2 or 3")

        self.conv1 = nn.Sequential(
            Conv(in_ch, in_ch, 3, padding=1, bias=False),
            BatchNorm(in_ch)
        )
        self.conv2 = nn.Sequential(
            Conv(in_ch, out_ch, 3, padding=1, bias=False),
            BatchNorm(out_ch)
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x  = self.conv1(x)
        x  = self.relu(x)
        x  = self.conv2(x)
        x  = self.relu(x)
        return x


class _ResDualConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dim: int) -> None:
        super(_ResDualConv, self).__init__()
        if   dim == 2: Conv, BatchNorm = nn.Conv2d, nn.BatchNorm2d
        elif dim == 3: Conv, BatchNorm = nn.Conv3d, nn.BatchNorm3d
        else: raise ValueError("dim must be 2 or 3")

        self.conv1 = nn.Sequential(
            Conv(in_ch, in_ch, 3, padding=1, bias=False),
            BatchNorm(in_ch)
        )
        self.conv2 = nn.Sequential(
            Conv(in_ch, out_ch, 3, padding=1, bias=False),
            BatchNorm(out_ch)
        )
        self.relu = nn.ReLU()

        self.skip = nn.Sequential(
            Conv(in_ch, out_ch, 1, bias=False),
            BatchNorm(out_ch),
        )
        if in_ch == out_ch: self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x  = self.conv1(x)
        x  = self.relu(x)
        x  = self.conv2(x)
        x += self.skip(residual)
        x  = self.relu(x)
        return x


class _UpConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dim: int) -> None:
        super(_UpConv, self).__init__()
        if   dim == 2: Conv, BatchNorm = nn.Conv2d, nn.BatchNorm2d
        elif dim == 3: Conv, BatchNorm = nn.Conv3d, nn.BatchNorm3d
        else: raise ValueError("dim must be 2 or 3")

        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(in_ch, out_ch, 3, padding=1, bias=False),
            BatchNorm(out_ch),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.upconv(x)


class _ChannelAttentionModule(nn.Module):
    def __init__(self, in_ch, dim: int, ratio: int = 16) -> None:
        super(_ChannelAttentionModule, self).__init__()
        if   dim == 2: 
            Conv = nn.Conv2d
            AdaptiveAvgPool = nn.AdaptiveAvgPool2d
            AdaptiveMaxPool = nn.AdaptiveMaxPool2d
        elif dim == 3:
            Conv = nn.Conv3d
            AdaptiveAvgPool = nn.AdaptiveAvgPool3d
            AdaptiveMaxPool = nn.AdaptiveMaxPool3d
        else: raise ValueError("dim must be 2 or 3")

        self.avg_pool = AdaptiveAvgPool(1)
        self.max_pool = AdaptiveMaxPool(1)
        self.MLP = nn.Sequential(
            Conv(in_ch, in_ch // ratio, 1, bias=False),
            nn.ReLU(),
            Conv(in_ch // ratio, in_ch, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.sigmoid(
            self.MLP(self.avg_pool(x)) + self.MLP(self.max_pool(x))
        )


class _SpatialAttentionModule(nn.Module):
    def __init__(self, dim: int) -> None:
        super(_SpatialAttentionModule, self).__init__()
        if   dim == 2: Conv = nn.Conv2d
        elif dim == 3: Conv = nn.Conv3d
        else: raise ValueError("dim must be 2 or 3")

        self.conv = Conv(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class _CBAM(nn.Module):
    def __init__(self, in_ch: int, dim: int) -> None:
        super(_CBAM, self).__init__()
        self.channel_attention = _ChannelAttentionModule(in_ch, dim)
        self.spatial_attention = _SpatialAttentionModule(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x


class _OutConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dim: int) -> None:
        super(_OutConv, self).__init__()
        if   dim == 2: Conv = nn.Conv2d
        elif dim == 3: Conv = nn.Conv3d
        else: raise ValueError("dim must be 2 or 3")

        self.outconv   = nn.Sequential(
            Conv(in_ch, out_ch, 1),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
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
            _CBAM(self.feats[i], self.dim)
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
