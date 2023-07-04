import torch
import numpy as np
from tifffile import imsave
import time


from config import Config as Config
from model import UNet2D, Criterion
from data import getDataLoader


if __name__ == "__main__":
    # configurations
    device = torch.device('cuda')
    config = Config()
    config.cpt_load_path = "checkpoints/test_6"
    config.batch_size = 10
    config.num_workers = 5
    config.num = [2000]
    config.type = ["Raw"]

    # load model
    net = UNet2D(config).to(device)  # config not important since load
    net.load_state_dict(torch.load(
        "{}.pt".format(config.cpt_load_path), 
        map_location=device)['net']
    )
    net.eval().half()

    # dataloader
    dataloader = getDataLoader(config)[0]

    # kernel
    kernel = Criterion.gaussianKernel(3).half().to(device)

    # eval
    with torch.no_grad():
        start = time.time()
        for i, (frames, labels) in enumerate(dataloader):
            # forward
            frames = frames.half().to(device)
            labels = labels.half().to(device)

            outputs = Criterion.gaussianBlur3d(net(frames), kernel)
        print(time.time()-start)
