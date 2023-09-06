import torch
import torch.backends.cudnn

import os
import tifffile
import tqdm

import config, model, data


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class Eval:
    def __init__(self, config) -> None:
        self.device = config.device
        self.ckpt_load_path   = config.ckpt_load_path
        self.data_save_folder = config.data_save_folder
        self.eval_type = config.eval_type
        self.lum_info  = config.lum_info

        # model
        self.model = model.ResAttUNet(config).to(self.device)
        self.model.load_state_dict(torch.load(
            "{}.ckpt".format(self.ckpt_load_path), 
            map_location=self.device)['model']
        )
        self.model.half()
        # data
        self.dataloader = data.getDataLoader(config)[0]
        self.dataset = self.dataloader.dataset  # to call static method

        # data index
        self.num_sub = self.dataset.num_sub  # type: ignore
        self.batch_size = self.dataloader.batch_size
        if self.num_sub % self.batch_size != 0:
            raise ValueError(
                "num_sub must be divisible by batch_size, but got {} and {}"
                .format(self.num_sub, self.batch_size)
            )

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
            total=int(
                len(self.dataloader)*self.batch_size/self.num_sub # type: ignore
            ),
            desc=self.data_save_folder, unit="frame"
        )

        # create folder
        if not os.path.exists(self.data_save_folder):
            os.makedirs(self.data_save_folder)

        sub_cat = None  # after concatenation
        sub_cmb = None  # after combination
        for i, (frames, labels) in enumerate(self.dataloader):
            if self.eval_type != "outputs" and self.eval_type != "labels":
                raise ValueError(
                    "eval_type must be 'outputs' or 'labels', but got {}"
                    .format(self.eval_type)
                )
            elif self.eval_type == "outputs":
                frames = frames.half().to(self.device)
                sub = self.model(frames)    # output of the model
            elif self.eval_type == "labels":
                sub = labels
            
            # since the both labels and outputs of model is binary result,
            # using self.lum_info to decide whether give brightness information
            if self.lum_info: sub = sub * frames

            # store subframes to a [self.num_sub, *output.shape] tensor
            sub_cat = sub if sub_cat is None else torch.cat((sub_cat, sub))

            # combine self.num_sub subframes to a [*output.shape] frame
            if len(sub_cat) < self.num_sub: continue
            if sub_cmb == None: 
                sub_cmb = self.dataset.combineFrame(sub_cat) # type: ignore
            else:
                sub_cmb += self.dataset.combineFrame(sub_cat) # type: ignore
            sub_cat = None

            # save after combine 1, 2, 4, 8, 16... frames
            current_frame = int((i+1)/(self.num_sub/self.batch_size))
            if current_frame & (current_frame - 1) == 0 \
            or i == len(self.dataloader) - 1:
                if self.eval_type == "outputs": tifffile.imwrite(
                    "{}/{:05}.tif".format(self.data_save_folder, current_frame),
                    sub_cmb.float().cpu().detach().numpy()
                )
                if self.eval_type == "labels": tifffile.imwrite(
                    "{}/{:05}.tif".format(self.data_save_folder, current_frame), 
                    sub_cmb.numpy()
                )

            pbar.update()  # update progress bar


if __name__ == "__main__":
    for cfg in config.getConfig():
        cfg.eval()
        evaluator = Eval(cfg)
        evaluator.eval()
