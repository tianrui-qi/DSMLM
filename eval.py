import torch

import os
from tqdm import tqdm
from tifffile import imsave

from config import getConfig
from model import UNet2D
from data import getDataLoader


class Eval:
    def __init__(self, config) -> None:
        # train
        self.device = config.device
        self.cpt_load_path = config.cpt_load_path
        # eval
        self.result_save_path = config.result_save_path
        # data
        self.num_sub = config.num_sub
        self.batch_size = config.batch_size

        # data
        self.dataloader = getDataLoader(config)[0]
        # model
        self.net = UNet2D(config).to(self.device)
        self.net.load_state_dict(torch.load(
            "{}.pt".format(self.cpt_load_path), 
            map_location=self.device)['net']
        )
        self.net.half()

        # record eval
        self.pbar = tqdm(
            total=len(self.dataloader) * self.batch_size / self.num_sub, 
            desc=self.cpt_load_path
        )
    
    @torch.no_grad()
    def eval(self):
        self.net.eval()
        outputs = None
        result = None
        for _, (frames, _) in enumerate(self.dataloader):
            # store subframe to a [100, *output.shape] tensor, i.e., outputs
            output = self.net(frames.half().to(self.device))
            if outputs is None: outputs = output
            else: outputs = torch.cat((outputs, output))

            # combine 100 subframe, i.e., outputs, to a frame
            if len(outputs) < self.num_sub: continue
            if result is None: result = \
                self.dataloader.dataset.combineFrame(outputs) # type: ignore
            else: result += \
                self.dataloader.dataset.combineFrame(outputs) # type: ignore
            outputs = None
            
            # update the progress bar every frame
            self.pbar.update()
        result /= torch.max(result)  # type: ignore

        # save
        if not os.path.exists(os.path.dirname(self.result_save_path)):
            os.makedirs(os.path.dirname(self.result_save_path))
        imsave(
            "{}.tif".format(self.result_save_path), 
            (result.cpu().detach() * 255).to(torch.uint8).numpy()
        )


if __name__ == "__main__":
    trainer = Eval(getConfig("eval"))
    trainer.eval()
