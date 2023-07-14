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
        self.ckpt_load_path = config.ckpt_load_path
        # eval
        self.outputs_save_path = config.outputs_save_path
        # data
        self.num_sub = config.num_sub
        self.batch_size = config.batch_size

        # data
        self.dataloader = getDataLoader(config)[0]
        # model
        self.net = UNet2D(config).to(self.device)
        self.net.load_state_dict(torch.load(
            "{}.ckpt".format(self.ckpt_load_path), 
            map_location=self.device)['net']
        )
        self.net.half()
    
    @torch.no_grad()
    def eval(self) -> None:
        # progress bar
        pbar = tqdm(
            total=len(self.dataloader) * self.batch_size / self.num_sub, 
            desc=self.ckpt_load_path, unit="frame"
        )

        outputs_cat = None  # output after concatenation
        outputs_cmb = None  # output after combination

        self.net.eval()
        for _, (frames, labels) in enumerate(self.dataloader):
            # store subframes to a [self.num_sub, *output.shape]
            outputs = self.net(frames.half().to(self.device))
            if outputs_cat is None: outputs_cat = outputs
            else: outputs_cat = torch.cat((outputs_cat, outputs))

            # combine self.num_sub subframes to a frame
            if len(outputs_cat) < self.num_sub: continue
            if outputs_cmb is None: outputs_cmb = \
                self.dataloader.dataset.combineFrame(outputs_cat) # type: ignore
            else: outputs_cmb += \
                self.dataloader.dataset.combineFrame(outputs_cat) # type: ignore
            outputs_cat = None
            
            pbar.update()  # update progress bar
        outputs_cmb /= torch.max(outputs_cmb)  # type: ignore

        # save
        if not os.path.exists(os.path.dirname(self.outputs_save_path)):
            os.makedirs(os.path.dirname(self.outputs_save_path))
        imsave(
            "{}.tif".format(self.outputs_save_path), 
            (outputs_cmb.cpu().detach() * 255).to(torch.uint8).numpy()
        )


if __name__ == "__main__":
    trainer = Eval(getConfig("eval"))
    trainer.eval()
