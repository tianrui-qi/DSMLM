import torch
from torch import nn

class UNet2D(nn.Module):
    def __init__(self):
        super(UNet2D, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        
        self.encoder1 = conv_block(256, 512)
        self.pool     = nn.MaxPool2d(2)
        self.encoder2 = conv_block(512, 1024)
        self.upconv   = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.decoder1 = conv_block(1024, 512)
        self.output   = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU()
        )
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.pool(enc1)
        enc2 = self.encoder2(enc2)
        enc2 = self.upconv(enc2)
        dec2 = torch.cat((enc1, enc2), dim=1)
        dec2 = self.decoder1(dec2)
        return self.output(dec2)


class DeepSTORMLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, frame, label):
        return self.mse(frame, label) + torch.norm(label, p=1)
