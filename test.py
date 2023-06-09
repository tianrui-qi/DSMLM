import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tifffile import imsave

from config import Config
from dataset import SimDataset
from model import UNet2D


def frame_sum(frame, label, output):
    # input is C * H * W frame, label, output
    # stact, reszie
    frame = np.sum(frame.numpy(), axis=0)
    frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST)
    frame = frame * 255 / np.amax(frame)
    label = np.sum(label.numpy(), axis=0)
    label[label > 0] = 255
    output = np.sum(output.detach().numpy(), axis=0)
    output[output > 0] = 255

    # change color
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    label = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
    label[:, :, 1] = 0  # G
    label[:, :, 2] = 0  # B
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    output[:, :, 0] = 0  # R

    # image calculate, add
    frame_sum = cv2.add(cv2.add(frame, label), output)
    frame_sum = np.clip(frame_sum, 0, 255)

    return frame_sum

def test_epoch(frame, label, idx, kernel_size, sigma, total_epoch):
    # This function will store a total_epoch * label_frame tif that represent
    # the net's prediction of this frame in each epoch
    # We may still need loop for different net (control by kernel_size and 
    # sigma) and different frame (control by idx)
    # Input is 1 * C * H * W frame, label (branch size is 1), which is the 
    # output of a dataloader
    img = []
    for epoch in range(1, total_epoch+1, 1):
        device = torch.device('cpu')
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load model
        net    = UNet2D(Config()).to(device)  # config not important since load
        net.load_state_dict(torch.load(
            "checkpoints/{}-{}-{}.pt".format(kernel_size, sigma, epoch), 
            map_location=device)['net'])

        # put frames and labels in device
        frame = frame.to(torch.float32).to(device)  # 1 * C * H * W
        label = label.to(torch.float32).to(device)  # 1 * C * H * W

        # predict
        net.eval()
        output = net(frame)  # 1 * C * H * W

        # print prediction information
        print(len(torch.nonzero(output)))
        print(len(torch.nonzero(label)))

        # frame sum for this epoch prediction      
        img_sum = frame_sum(
            frame.reshape(frame.shape[1:]),     # C * H * W
            label.reshape(label.shape[1:]),     # C * H * W
            output.reshape(output.shape[1:])    # C * H * W
            )
        img.append(img_sum)
    imsave('tests/{}-{}-{}.tif'.format(idx, kernel_size, sigma), 
                np.array(img).astype(np.uint8))
        
if __name__ == "__main__":
    kernel_size = 9
    sigma = 2
    total_epoch = 1

    for seed in range(5):  # sample index
        np.random.seed(seed)
        validloader = DataLoader(SimDataset(Config(), 1),)
        for i, (frame, label) in enumerate(validloader): 
            test_epoch(frame, label, seed, kernel_size, sigma, total_epoch)
    