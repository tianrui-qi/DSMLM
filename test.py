import torch
from torch.utils.data import DataLoader
import numpy as np

from config import Config
from dataset import SimDataset
from model import UNet2D


# configuration
config = Config()
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load model
net    = UNet2D(config).to(device)
net.load_state_dict(torch.load("checkpoints/7-(5,5).pt", map_location=device)['net'])

np.random.seed(0)
validloader = DataLoader(
            SimDataset(config, 1),
            batch_size=config.batch_size, 
            #num_workers=config.batch_size, 
            #pin_memory=True
            )

net.eval()
for i, (frame, label) in enumerate(validloader):
    import cv2
    from tifffile import imsave

    # put frames and labels in GPU
    frame = frame.to(torch.float32).to(device)
    label = label.to(torch.float32).to(device)
    # forward
    output = net(frame)

    frame = frame.reshape(frame.shape[1:])
    label = label.reshape(label.shape[1:])
    output = output.reshape(output.shape[1:])

    print(len(torch.nonzero(output)))
    print(len(torch.nonzero(label)))

    # stact, reszie
    frame = np.sum(frame.numpy(), axis=0)
    frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST)
    frame = frame * 255 / np.amax(frame)
    label = np.sum(label.numpy(), axis=0)
    label[label > 0] = 255
    output = np.sum(output.detach().numpy(), axis=0)
    output[output > 0] = 255
    
    # save
    #imsave('frame.tif', frame.astype(np.uint8))
    #imsave('label.tif', label.astype(np.uint8))
    #imsave('output.tif', output.astype(np.uint8))

    # change color
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    label = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
    label[:, :, 1] = 0  # G
    label[:, :, 2] = 0  # B
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    output[:, :, 0] = 0  # R
    img_sum = cv2.add(cv2.add(frame, label), output)
    img_sum = np.clip(img_sum, 0, 255).astype(np.uint8)
    # save
    imsave('img_sum.tif', img_sum.astype(np.uint8))
