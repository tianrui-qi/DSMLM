import torch
import torch.backends.cudnn

import os
import tifffile
import tqdm

import sml.model
import sml.data


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class Evaluator:
    def __init__(self, config) -> None:
        self.device = config.device
        self.ckpt_load_path = config.ckpt_load_path
        self.data_save_fold = config.data_save_fold

        # model
        self.model = sml.model.ResAttUNet(config).to(self.device)
        self.model.load_state_dict(torch.load(
            "{}.ckpt".format(self.ckpt_load_path), 
            map_location=self.device)['model']
        )
        self.model.half()
        # data
        self.dataloader = sml.data.getDataLoader(config)[0]
        self.dataset = self.dataloader.dataset  # to call static method

        # data index
        self.num_sub    = self.dataset.num_sub
        self.batch_size = self.dataloader.batch_size
        if self.num_sub % self.batch_size != 0: raise ValueError(
            "num_sub must be divisible by batch_size, but got {} and {}"
            .format(self.num_sub, self.batch_size)
        )

        # print model info
        num_paras = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f'The model has {num_paras:,} trainable parameters')

    @torch.no_grad()
    def eval(self) -> None:
        self.model.eval()

        # record: progress bar
        pbar = tqdm.tqdm(
            total=int(len(self.dataloader)*self.batch_size/self.num_sub),
            desc=self.data_save_fold, unit="frame"
        )

        # create folder
        if not os.path.exists(self.data_save_fold):
            os.makedirs(self.data_save_fold)

        sub_cat = None  # after concatenation
        sub_cmb = None  # after combination
        for i, frames in enumerate(self.dataloader):
            frames = frames.half().to(self.device)
            sub = self.model(frames)    # prediction from the model

            # store subframes to a [self.num_sub, *sub.shape] tensor
            sub_cat = sub if sub_cat is None else torch.cat((sub_cat, sub))

            # combine self.num_sub number of subframes to a whole frames
            if len(sub_cat) < self.num_sub: continue
            if sub_cmb == None: sub_cmb = self.dataset.combineFrame(sub_cat)
            else: sub_cmb += self.dataset.combineFrame(sub_cat)
            sub_cat = None

            # save after combine 1, 2, 4, 8, 16... frames
            current_frame = int((i+1)/(self.num_sub/self.batch_size))
            if current_frame & (current_frame - 1) == 0 \
            or i == len(self.dataloader) - 1: tifffile.imwrite(
                "{}/{:05}.tif".format(self.data_save_fold, current_frame),
                sub_cmb.float().cpu().detach().numpy()
            )

            pbar.update()  # update progress bar
