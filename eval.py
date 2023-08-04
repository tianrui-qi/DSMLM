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
        self.ckpt_load_path = config.ckpt_load_path
        self.data_save_folder = config.data_save_folder

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

        outputs_cat = None  # output after concatenation
        outputs_cmb = None  # output after combination
        labels_cat  = None  # label after concatenation
        labels_cmb  = None  # label after combination

        for i, (frames, labels) in enumerate(self.dataloader):
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

            # save after combine 1, 2, 4, 8, 16... frame
            current_frame = int((i+1)/(self.num_sub/self.batch_size))
            if current_frame & (current_frame - 1) == 0 \
            or i == len(self.dataloader) - 1:
                tifffile.imwrite(
                    "{}/outputs_{}.tif".format(
                        self.data_save_folder, current_frame
                    ), outputs_cmb.float().cpu().detach().numpy()
                )
                tifffile.imwrite(
                    "{}/labels_{}.tif".format(
                        self.data_save_folder, current_frame
                    ), labels_cmb.numpy()
                )

            pbar.update()  # update progress bar


if __name__ == "__main__":
    for cfg in config.getConfig():
        cfg.eval()
        evaluator = Eval(cfg)
        evaluator.eval()
