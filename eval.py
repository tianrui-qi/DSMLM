import torch

import os
import tifffile
import tqdm

import config, data


class Eval:
    def __init__(self, config) -> None:
        # train
        self.device = config.device
        self.ckpt_load_path = config.ckpt_load_path
        # eval
        self.outputs_save_path = config.outputs_save_path
        self.labels_save_path = config.labels_save_path
        # data
        num_sub_h = config.h_range[1] - config.h_range[0] + 1
        num_sub_w = config.w_range[1] - config.w_range[0] + 1
        self.num_sub = num_sub_h * num_sub_w
        self.batch_size = config.batch_size

        # data
        self.dataloader = data.getDataLoader(config)[0]
        self.dataset = self.dataloader.dataset  # to call static method
        # model
        self.model = config.model(config).to(self.device)
        self.model.load_state_dict(torch.load(
            "{}.ckpt".format(self.ckpt_load_path), 
            map_location=self.device)['model']
        )
        self.model.half()

        # print model info
        para_num = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f'The model has {para_num:,} trainable parameters')
    
    @torch.no_grad()
    def eval(self) -> None:
        self.model.eval()

        # record: progress bar
        pbar = tqdm.tqdm(
            total=len(self.dataloader)*self.batch_size/self.num_sub, 
            desc=self.ckpt_load_path, unit="frame"
        )

        outputs_cat = None  # output after concatenation
        outputs_cmb = None  # output after combination
        labels_cat  = None  # label after concatenation
        labels_cmb  = None  # label after combination

        for _, (frames, labels) in enumerate(self.dataloader):
            # store subframes to a [self.num_sub, *output.shape] tensor
            outputs = self.model(frames.half().to(self.device))
            if outputs_cat == None: outputs_cat = outputs
            else: outputs_cat = torch.cat((outputs_cat, outputs))
            if labels_cat == None: labels_cat = labels
            else: labels_cat = torch.cat((labels_cat, labels))

            # combine self.num_sub subframes to a [*output.shape] frame
            if len(outputs_cat) < self.num_sub: continue
            if outputs_cmb == None: 
                outputs_cmb = self.dataset.combineFrame(outputs_cat) # type: ignore
            else:
                outputs_cmb += self.dataset.combineFrame(outputs_cat) # type: ignore
            if labels_cmb == None:
                labels_cmb = self.dataset.combineFrame(labels_cat)  # type: ignore
            else:
                labels_cmb += self.dataset.combineFrame(labels_cat)  # type: ignore
            outputs_cat = None
            labels_cat = None

            pbar.update()  # update progress bar

        # save
        if not os.path.exists(os.path.dirname(self.outputs_save_path)):
            os.makedirs(os.path.dirname(self.outputs_save_path))
        if not os.path.exists(os.path.dirname(self.labels_save_path)):
            os.makedirs(os.path.dirname(self.labels_save_path))
        tifffile.imsave(
            "{}.tif".format(self.outputs_save_path),
            outputs_cmb.cpu().detach().numpy()  # type: ignore
        )
        tifffile.imsave(
            "{}.tif".format(self.labels_save_path), 
            labels_cmb.numpy()  # type: ignore
        )


if __name__ == "__main__":
    trainer = Eval(config.getConfig("eval"))
    trainer.eval()
