import os
import torch
import numpy as np
import cv2
from tifffile import imsave


from data import getDataLoader
from model import UNet2D


def frame_sum(frame, label, output):
    # input is C * H * W frame, label, output
    # stact, reszie
    label = np.sum(label.cpu().numpy(), axis=0)
    if np.amax(label) != 0: label = label * 255 / np.amax(label)
    output = np.sum(output.detach().cpu().numpy(), axis=0)
    if np.amax(output) != 0: output = output * 255 / np.amax(output)
    output[output > 0] = 255
    frame = np.sum(frame.cpu().numpy(), axis=0)
    frame = cv2.resize(frame, label.shape, interpolation=cv2.INTER_NEAREST)
    if np.amax(frame) != 0: frame = frame * 128 / np.amax(frame) # type: ignore
    
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
    from config import Test_1 as Config
    config = Config()
    config.batch_size = 1
    config.num_workers = 1

    dataloader = getDataLoader(config)[1]
    for i, (frame, label) in enumerate(dataloader): 
        if i == 20: break

        device = torch.device('cpu')
        load_dir = "checkpoints/test_1"
        save_dir = "data/test_1/{}.tif".format(i)
        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))

        test_epochs(frame, label, config, device, load_dir, save_dir)
