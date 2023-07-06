import torch
from tifffile import imsave

from config import Eval
from model import UNet2D, Criterion
from data import getDataLoader


if __name__ == "__main__":
    # configurations
    config = Eval()
    device = torch.device('cuda')

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
        outputs = None
        for i, (frames, labels) in enumerate(dataloader):
            # forward
            frames = frames.half().to(device)
            labels = labels.half().to(device)

            output = Criterion.gaussianBlur3d(net(frames), kernel)

            if outputs == None: outputs = output
            else: outputs = torch.cat((outputs, output))
            
            if len(outputs) == 100:
                frame = dataloader.dataset.combineFrame(outputs) # type: ignore
                imsave(
                    'data/eval/frame.tif', 
                    (frame.detach() * 255).to(torch.uint8).numpy()
                )
                outputs = None
