import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class UNetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(UNetBlock2D, self).__init__()
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


class UNetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(UNetBlock3D, self).__init__()
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


class ResUNetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUNetBlock2D, self).__init__()
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

    def forward(self, x):
        residual = x
        out  = self.conv1(x)
        out  = self.relu(out)
        out  = self.conv2(out)
        out += self.skip(residual)
        out  = self.relu(out)
        return out


class ResUNetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUNetBlock3D, self).__init__()
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

    def forward(self, x):
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
        in_feature = config.dim_frame[0]  # input feature/channel/depth num
        up_c       = config.up_sample[0]  # upsampling scale, channel/depth

        self.input    = nn.Upsample(scale_factor=tuple(config.up_sample))
        self.encoder1 = ResUNetBlock2D(in_feature*up_c*1, in_feature*up_c*2)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.encoder2 = ResUNetBlock2D(in_feature*up_c*2, in_feature*up_c*4)
        self.up_conv  = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_feature*up_c*4, in_feature*up_c*2, 3, padding=1)
        )
        self.decoder1 = ResUNetBlock2D(in_feature*up_c*4, in_feature*up_c*2)
        self.output   = nn.Sequential(
            nn.Conv2d(in_feature * up_c * 2, in_feature * up_c, 1),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        enc1 = self.input(x.unsqueeze(1)).squeeze(1)
        enc1 = self.encoder1(enc1)
        enc2 = self.max_pool(enc1)
        enc2 = self.encoder2(enc2)
        enc2 = self.up_conv(enc2)
        dec2 = torch.cat((enc1, enc2), dim=1)
        dec2 = self.decoder1(dec2)
        return self.output(dec2)


class Criterion(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # Gaussian kernel using help function gaussianKernel
        self.kernel = self.gaussianKernel(
            3, config.kernel_size, config.kernel_sigma)
        # pad size, pad before convolve Gaussian kernel
        self.pad = [config.kernel_size for _ in range(6)]  # [C H W]

    def forward(self, predi: Tensor, label: Tensor) -> float:
        return F.mse_loss(
            self.gaussianBlur3d(
                F.pad(predi, self.pad, mode='reflect'), self.kernel),
            self.gaussianBlur3d(
                F.pad(label, self.pad, mode='reflect'), self.kernel),
            reduction="sum"
        )  # type: ignore

    def to(self, device):
        # Call the original 'to' method to move parameters and buffers
        super(Criterion, self).to(device)
        self.kernel = self.kernel.to(device)
        return self

    @staticmethod
    def gaussianKernel(
        dim: int, 
        kernel_size: int = 7, kernel_sigma: float = 1.0
    ) -> Tensor:
        """
        This function generates a Gaussian kernel with the given parameters. The
        function will first build a 1D Gaussian kernel where the size and sigma
        are controlled by `kernel_size` and `kernel_sigma`. Then, we extend it
        to nd Gaussian kernel using the einsum function where nd is specified by
        `dim`.

        Args:
            dim (int): The dimension of the kernel. For example, set dim=2 for a 
                2D Gaussian kernel.
            kernel_size (int): The size of the kernel, usually an odd number
                like 3, 5, 7, etc. Default: 7.
            kernel_sigma (float): The sigma of the Gaussian kernel. 
                Default: 1.0.
        
        Returns:
            kernel_nd (Tensor): A Gaussian kernel with shape 
                (kernel_size,) * dim.
        """
        # build 1D Gaussian kernel
        r = kernel_size // 2
        coord = torch.linspace(-r, r, kernel_size)
        kernel_1d = torch.exp((-(coord / kernel_sigma) ** 2 / 2))
        kernel_1d /= kernel_1d.sum()  # Normalization

        # build nd Gaussian kernel using einsum
        equation = ','.join(f'{chr(97 + i)}' for i in range(dim))
        operands = [kernel_1d for _ in range(dim)]
        kernel_nd  = torch.einsum(equation, *operands)
        kernel_nd /= kernel_nd.sum()  # Normalization

        return kernel_nd

    @staticmethod
    def gaussianBlur3d(frame: Tensor, kernel: Tensor) -> Tensor:
        """
        This function convolve the input frame with a 3D Gaussian kernel using
        F.conv3d. We accept input frame with shape [D H W], [B C H W], or 
        [B C D H W] and will convolve last three dimension. If the shape is 
        [B C H W], we treat the channel as the depth. The input kernel should 
        shape like [D H W].

        Args:
            frame (Tensor): The frame to be convolved with the Gaussian kernel
                with shape [D H W], [B C H W], or [B C D H W].
            kernel (Tensor): The Gaussian kernel with shape [D H W].

        Returns:
            frame_blur (Tensor): The blurred frame with the same shape as the
                input frame.

        Raises:
            ValueError: If the dim of the kernel is not 3.
            ValueError: If the dim of the frame is not 3, 4 or 5.
        """
        if kernel.dim() != 3:
            raise ValueError("kernel.dim() must be 3")
        kernel = kernel.reshape(1, 1, *kernel.shape)  # [D H W] -> [1 1 D H W]

        # set the shape of the frame when convolve with the kernel
        if frame.dim() == 3:    # [D H W] -> [1 1 D H W]
            shape = [1, 1, *frame.shape]
        elif frame.dim() == 4:  # [B C H W] -> [B 1 C H W]
            shape = [frame.shape[0], 1, *frame.shape[1:]]
        elif frame.dim() == 5:  # [B C D H W]
            shape = frame.shape
        else:
            raise ValueError("frame.dim() must be 3, 4 or 5")

        return F.conv3d(
                frame.reshape(*shape), kernel, padding="same"
            ).reshape(*frame.shape)
