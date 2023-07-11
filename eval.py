import torch
from tqdm import tqdm
from tifffile import imsave

from config import ConfigEval
from model import UNet2D
from data import getDataLoader


if __name__ == "__main__":
    # config
    config = ConfigEval()
    device = torch.device('cuda')

    # model
    net = UNet2D(config).to(device)  # config not important since load
    net.load_state_dict(torch.load(
        "{}.pt".format(config.cpt_load_path), 
        map_location=device)['net']
    )
    net.eval().half()

    # data
    dataloader = getDataLoader(config)[0]

    # eval
    with torch.no_grad():
        outputs = None
        frame = None
        for i, (frames, _) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Processing"):
            # store subframe to a [100, *output.shape] tensor, i.e., outputs
            output = net(frames.half().to(device))
            if outputs == None: outputs = output
            else: outputs = torch.cat((outputs, output))

            # combine 100 subframe, i.e., outputs, to a frame
            if len(outputs) != 100: continue
            if frame == None: 
                frame  = dataloader.dataset.combineFrame(outputs) # type: ignore
            else:
                frame += dataloader.dataset.combineFrame(outputs) # type: ignore
            outputs = None

    # save
    frame /= torch.max(frame)  # type: ignore
    imsave(
        'data/30.tif', (frame.cpu().detach() * 255).to(torch.uint8).numpy()
    )
