import torch
from torch.utils.data import DataLoader
import numpy as np

from config import Config
from dataset import SimDataset
from model import UNet2D


# configuration
config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load model
net    = UNet2D().to(device)
net.load_state_dict(torch.load(config.checkpoint_path)['net'])


validloader = DataLoader(
            SimDataset(config, 2),
            batch_size=config.batch_size, 
            num_workers=config.batch_size, 
            pin_memory=True
            )

net.eval()
for i, (frames, labels) in enumerate(validloader):
    # put frames and labels in GPU
    frames = frames.to(torch.float32).to(device)
    labels = labels.to(torch.float32).to(device)
    # forward
    outputs = net(frames)

    print(torch.nonzero(outputs))
    print(torch.nonzero(labels))
