import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(UNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UNet2D(nn.Module):
    def __init__(self, config) -> None:
        super(UNet2D, self).__init__()

        in_feature = config.dim_frame[0]  # input feature/channel/depth num
        up_c = config.up_sample[0]  # upsampling scale, channel/depth

        self.input = nn.Upsample(
            scale_factor=tuple(config.up_sample), mode='nearest')

        self.encoder1 = UNetBlock(in_feature * up_c * 1, in_feature * up_c * 2)
        self.pool = nn.MaxPool2d(2)
        self.encoder2 = UNetBlock(in_feature * up_c * 2, in_feature * up_c * 4)
        self.upconv = nn.ConvTranspose2d(
            in_feature * up_c * 4, in_feature * up_c * 2, 4, stride=2, padding=1)
        self.decoder1 = UNetBlock(in_feature * up_c * 4, in_feature * up_c * 2)

        self.output = nn.Sequential(
            nn.Conv2d(in_feature * up_c * 2, in_feature * up_c, 1),
            nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        up = self.input(x.unsqueeze(1)).squeeze(1)
        # up   = self.input(x)
        enc1 = self.encoder1(up)
        enc2 = self.pool(enc1)
        enc2 = self.encoder2(enc2)
        enc2 = self.upconv(enc2)
        dec2 = torch.cat((enc1, enc2), dim=1)
        dec2 = self.decoder1(dec2)
        return self.output(dec2)


class Criterion(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        # configuration
        self.kernel_size = config.kernel_size
        self.kernel_sigma = config.kernel_sigma
        self.l1_coeff = config.l1_coeff

        # Gaussian kernel using help function generateGaussianKernel
        self.kernel = self.generateGaussianKernel(
            3, self.kernel_size, self.kernel_sigma)

        # pad size, pad before convolve Gaussian kernel
        self.pad = [self.kernel_size for _ in range(6)]  # [C H W]

    def forward(self, predi: Tensor, label: Tensor) -> float:
        mse_loss = F.mse_loss(
            self.generateGaussianBlur3d(
                F.pad(predi, self.pad, mode='reflect'), self.kernel),
            self.generateGaussianBlur3d(
                F.pad(label, self.pad, mode='reflect'), self.kernel),
            reduction="sum"
        )
        l1_loss = F.l1_loss(predi, torch.zeros_like(predi), reduction="sum")
        
        return mse_loss + self.l1_coeff * l1_loss

    def to(self, device):
        # Call the original 'to' method to move parameters and buffers
        super(Criterion, self).to(device)
        self.kernel = self.kernel.to(device)
        return self

    @staticmethod
    def generateGaussianKernel(
        dim: int, kernel_size: int = 7, kernel_sigma: float = 1.0
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
            kernel_sigma (float): The sigma of the Gaussian kernel. Default: 1
        
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
        kernel_nd = torch.einsum(equation, *operands)
        kernel_nd /= kernel_nd.sum()  # Normalization

        return kernel_nd

    @staticmethod
    def generateGaussianBlur3d(frame: Tensor, kernel: Tensor) -> Tensor:
        """
        This function convolve a frame with a 3D Gaussian kernel using
        F.conv3d. We accept frame with shape [B C H W] and [B C D H W]. If the
        shape is [B C H W], we treat the channel as the depth. The kernel should
        shape like [D H W]. We will reshape the kernel to [1 1 D H W] before
        call the F.conv3d function.

        Args:
            frame (Tensor): The frame to be convolved with the Gaussian kernel
                with shape [B C H W] or [B C D H W].
            kernel (Tensor): The Gaussian kernel with shape [D H W].

        Returns:
            frame_blur (Tensor): The blurred frame with the same shape as the
                input frame.

        Raises:
            ValueError: If the dim of the kernel is not 3.
            ValueError: If the dim of the frame is not 4 or 5.
        """
        if kernel.dim() != 3:
            raise ValueError("kernel.dim() must be 3")
        kernel = kernel.reshape(1, 1, *kernel.shape)  # [D H W] to  [1 1 D H W]

        if frame.dim() == 4:
            return F.conv3d(
                frame.unsqueeze(1), kernel, padding="same"
            ).reshape(*frame.shape)
        elif frame.dim() == 5:
            return F.conv3d(frame, kernel, padding="same")
        else:
            raise ValueError("frame.dim() must be 4 or 5")
