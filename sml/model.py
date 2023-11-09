import torch
import torch.nn as nn
from torch import Tensor

from typing import List


class _ChannelAttentionModule(nn.Module):
    def __init__(self, dim: int, in_c: int, ratio: int = 16) -> None:
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
            Conv(in_c, in_c // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            Conv(in_c // ratio, in_c, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.sigmoid(
            self.MLP(self.avg_pool(x)) + self.MLP(self.max_pool(x))
        )


class _SpatialAttentionModule(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7) -> None:
        super(_SpatialAttentionModule, self).__init__()
        if   dim == 2: Conv = nn.Conv2d
        elif dim == 3: Conv = nn.Conv3d
        else: raise ValueError("dim must be 2 or 3")

        self.conv = Conv(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class _DualConv(nn.Module):
    def __init__(
        self, dim: int, in_c: int, out_c: int, 
        use_cbam: bool = False, use_res: bool = False
    ) -> None:
        super(_DualConv, self).__init__()
        self.use_cbam = use_cbam
        self.use_res  = use_res

        if   dim == 2: Conv, BatchNorm = nn.Conv2d, nn.BatchNorm2d
        elif dim == 3: Conv, BatchNorm = nn.Conv3d, nn.BatchNorm3d
        else: raise ValueError("dim must be 2 or 3")

        # dual convolution
        self.conv1 = nn.Sequential(
            Conv(in_c, out_c, 3, padding=1, bias=False),
            BatchNorm(out_c)
        )
        self.conv2 = nn.Sequential(
            Conv(out_c, out_c, 3, padding=1, bias=False),
            BatchNorm(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

        # cbam
        if self.use_cbam:
            self.channel_attention = _ChannelAttentionModule(dim, out_c)
            self.spatial_attention = _SpatialAttentionModule(dim)

        # residual
        if self.use_res:
            if in_c == out_c:
                self.skip = nn.Identity()
            else:
                self.skip = nn.Sequential(
                    Conv(in_c, out_c, 1, bias=False),
                    BatchNorm(out_c),
                )

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res: 
            res = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.use_cbam: 
            x = self.channel_attention(x) * x
            x = self.spatial_attention(x) * x
        if self.use_res: 
            x += self.skip(res)  # type: ignore
        return self.relu(x)


class ResAttUNet(nn.Module):
    def __init__(
        self, dim: int, feats: List[int], use_cbam: bool, use_res: bool
    ) -> None:
        super(ResAttUNet, self).__init__()
        self.dim      = dim
        self.feats    = feats
        self.use_cbam = use_cbam
        self.use_res  = use_res

        if   self.dim == 2: Conv, MaxPool = nn.Conv2d, nn.MaxPool2d
        elif self.dim == 3: Conv, MaxPool = nn.Conv3d, nn.MaxPool3d
        else: raise ValueError("dim must be 2 or 3")

        # encoder
        self.encoder = nn.ModuleList([
            _DualConv(
                self.dim, self.feats[i], self.feats[i+1], 
                self.use_cbam, self.use_res
            ) for i in range(len(self.feats)-1)
        ])
        self.maxpool = nn.ModuleList([
            MaxPool(2) for _ in range(len(self.feats)-2)
        ])
        # decoder
        self.upconv = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                Conv(self.feats[i+1], self.feats[i], 1),
            ) for i in range(1, len(self.feats)-1)
        ])
        self.decoder = nn.ModuleList([
            _DualConv(
                self.dim, self.feats[i+1], self.feats[i], 
                self.use_cbam, self.use_res
            ) for i in range(1, len(self.feats)-1)
        ])
        # output
        self.outconv = nn.Sequential(
            Conv(self.feats[1], self.feats[0], 1),
            nn.ReLU(inplace=True)
        )

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
            # upconv
            x = self.upconv[-i](x)
            # decoder
            x = self.decoder[-i](torch.cat((enc[-i], x), dim=1))
        # output
        x = self.outconv(x)

        if self.dim == 3: x = x.squeeze(1)
        return x
