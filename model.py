import torch
import torch.nn as nn
from torch import Tensor


class _DualConv(nn.Module):
    def __init__(
        self, dim: int, in_c: int, out_c: int, use_res: bool = True
    ) -> None:
        super(_DualConv, self).__init__()
        self.use_res = use_res

        if   dim == 2: Conv, BatchNorm = nn.Conv2d, nn.BatchNorm2d
        elif dim == 3: Conv, BatchNorm = nn.Conv3d, nn.BatchNorm3d
        else: raise ValueError("dim must be 2 or 3")

        self.conv1 = nn.Sequential(
            Conv(in_c, in_c, 3, padding=1, bias=False),
            BatchNorm(in_c)
        )
        self.conv2 = nn.Sequential(
            Conv(in_c, out_c, 3, padding=1, bias=False),
            BatchNorm(out_c)
        )
        self.relu = nn.ReLU()

        if not self.use_res: return
        self.skip = nn.Sequential(
            Conv(in_c, out_c, 1, bias=False),
            BatchNorm(out_c),
        )
        if in_c == out_c: self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res: res = x
        x  = self.conv1(x)
        x  = self.relu(x)
        x  = self.conv2(x)
        if self.use_res: x += self.skip(res)  # type: ignore
        x  = self.relu(x)
        return x


class _UpConv(nn.Module):
    def __init__(self, dim: int, in_c: int, out_c: int) -> None:
        super(_UpConv, self).__init__()
        if   dim == 2: Conv, BatchNorm = nn.Conv2d, nn.BatchNorm2d
        elif dim == 3: Conv, BatchNorm = nn.Conv3d, nn.BatchNorm3d
        else: raise ValueError("dim must be 2 or 3")

        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(in_c, out_c, 3, padding=1, bias=False),
            BatchNorm(out_c),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.upconv(x)


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
            nn.ReLU(),
            Conv(in_c // ratio, in_c, 1, bias=False)
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
    def __init__(self, dim: int, in_c: int, use_res: bool = True) -> None:
        super(_CBAM, self).__init__()
        self.use_res = use_res

        self.channel_attention = _ChannelAttentionModule(dim, in_c)
        self.spatial_attention = _SpatialAttentionModule(dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res: res = x
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        if self.use_res: x += res  # type: ignore
        return x


class _AttGate(nn.Module):
    def __init__(self, dim: int, in_c: int, n_coeff: int) -> None:
        super(_AttGate, self).__init__()
        if   dim == 2: Conv, BatchNorm = nn.Conv2d, nn.BatchNorm2d
        elif dim == 3: Conv, BatchNorm = nn.Conv3d, nn.BatchNorm3d
        else: raise ValueError("dim must be 2 or 3")

        self.w_decod = nn.Sequential(
            Conv(in_c, n_coeff, 1, bias=False),
            BatchNorm(n_coeff)
        )
        self.w_encod = nn.Sequential(
            Conv(in_c, n_coeff, 1, bias=False),
            BatchNorm(n_coeff)
        )
        self.relu = nn.ReLU()
        self.psi = nn.Sequential(
            Conv(n_coeff, 1, 1, bias=False),
            BatchNorm(1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor, decod: Tensor) -> Tensor:
        psi = self.w_encod(x) + self.w_decod(decod)
        psi = self.relu(psi)
        psi = self.psi(psi)
        return x * psi


class _OutConv(nn.Module):
    def __init__(self, dim: int, in_c: int, out_c: int) -> None:
        super(_OutConv, self).__init__()
        if   dim == 2: Conv = nn.Conv2d
        elif dim == 3: Conv = nn.Conv3d
        else: raise ValueError("dim must be 2 or 3")

        self.outconv   = nn.Sequential(
            Conv(in_c, out_c, 1),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.outconv(x)


class ResAttUNet(nn.Module):
    def __init__(self, config) -> None:
        super(ResAttUNet, self).__init__()
        self.dim   = config.dim
        self.feats = config.feats
        self.use_res  = config.use_res
        self.use_cbam = config.use_cbam
        self.use_att  = config.use_att

        if   self.dim == 2: MaxPool = nn.MaxPool2d
        elif self.dim == 3: MaxPool = nn.MaxPool3d
        else: raise ValueError("dim must be 2 or 3")

        # encoder
        self.encoder = nn.ModuleList([
            _DualConv(self.dim, self.feats[i], self.feats[i+1], self.use_res) 
            for i in range(len(self.feats)-1)
        ])
        self.maxpool = nn.ModuleList([
            MaxPool(2) 
            for _ in range(len(self.feats)-2)
        ])
        # decoder
        self.upconv = nn.ModuleList([
            _UpConv(self.dim, self.feats[i+1], self.feats[i])
            for i in range(1, len(self.feats)-1)
        ])
        if self.use_cbam: self.cbam = nn.ModuleList([
            _CBAM(self.dim, self.feats[i], self.use_res)
            for i in range(1, len(self.feats)-1)
        ])
        if self.use_att: self.attgate = nn.ModuleList([
            _AttGate(self.dim, self.feats[i], self.feats[i]//2)
            for i in range(1, len(self.feats)-1)
        ])
        self.decoder = nn.ModuleList([
            _DualConv(self.dim, self.feats[i+1], self.feats[i], self.use_res)
            for i in range(1, len(self.feats)-1)
        ])
        # output
        self.outconv = _OutConv(self.dim, self.feats[1], self.feats[0])

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
            # cbam
            if self.use_cbam: enc[-i] = self.cbam[-i](enc[-i])
            # attention gate
            if self.use_att:  enc[-i] = self.attgate[-i](enc[-i], x)
            # decoder
            x = self.decoder[-i](torch.cat((enc[-i], x), dim=1))
        # output
        x = self.outconv(x)

        if self.dim == 3: x = x.squeeze(1)
        return x
