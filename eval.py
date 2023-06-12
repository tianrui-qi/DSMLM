import os
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
    frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_NEAREST)
    frame = frame * 128 / np.amax(frame)
    label = np.sum(label.numpy(), axis=0)
    if np.amax(label) != 0: label = label * 255 / np.amax(label)
    #label[label > 0] = 255
    output = np.sum(output.detach().numpy(), axis=0)
    if np.amax(output) != 0: output = output * 255 / np.amax(output)
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

def test_epoch(frame, label, net):
    # predict
    net.eval()
    output = net(frame)  # 1 * C * H * W

    # print prediction information
    print(len(torch.nonzero(output)))
    print(len(torch.nonzero(label)))

    return frame_sum(
        frame.reshape(frame.shape[1:]),     # C * H * W
        label.reshape(label.shape[1:]),     # C * H * W
        output.reshape(output.shape[1:])    # C * H * W
        )  # H * W

def test_epochs(frame, label, config, device, load_dir, save_dir):
    num_epoch= len([f for f in os.listdir(load_dir) if os.path.isfile(
        os.path.join(load_dir, f))])

    img = []
    for epoch in range(num_epoch, 0, -1):
        # load model
        net = UNet2D(config).to(device)  # config not important since load
        net.load_state_dict(torch.load(
            os.path.join(load_dir, "{}.pt".format(epoch)), 
            map_location=device)['net'])

        # put frames and labels in device
        frame = frame.to(torch.float32).to(device)  # 1 * C * H * W
        label = label.to(torch.float32).to(device)  # 1 * C * H * W

        # test current epoch
        img.append(test_epoch(frame, label, net))

    imsave(save_dir, np.array(img).astype(np.uint8))

        
if __name__ == "__main__":
    for kernel_sigma in [1]:
        config = Config()
        config.dim_frame = [32, 32, 32]
        config.up_sample = [2, 2, 2]

        for seed in range(3):  # sample index
            np.random.seed(seed)
            validloader = DataLoader(SimDataset(config, 1),)
            for i, (frame, label) in enumerate(validloader): 
                device = torch.device('cpu')

                load_dir = "checkpoints/exp2"
                save_dir = "assets/exp2/{}-7-{}.tif".format(seed, kernel_sigma)
                if not os.path.exists(os.path.dirname(save_dir)):
                    os.makedirs(os.path.dirname(save_dir))

                test_epochs(frame, label, config, device, load_dir, save_dir)
    exit()
    for kernel_size in [3, 5, 7, 9, 11]:
        config = Config()
        config.dim_frame = [32, 32, 32]
        config.up_sample = [2, 2, 2]

        for seed in range(3):  # sample index
            np.random.seed(seed)
            validloader = DataLoader(SimDataset(config, 1),)
            for i, (frame, label) in enumerate(validloader): 
                device = torch.device('cpu')

                load_dir = "checkpoints/test_criterion_kernel_size/{}-1".format(kernel_size)
                save_dir = "assets/test_criterion_kernel_size/{}-{}-1.tif".format(seed, kernel_size)
                if not os.path.exists(os.path.dirname(save_dir)):
                    os.makedirs(os.path.dirname(save_dir))

                test_epochs(frame, label, config, device, load_dir, save_dir)
