import torch
import torch.utils.data                 # DataLoader
import torch.nn.functional as F
from torch import Tensor

import numpy as np

import os
import tqdm
import tifffile

import src.data, src.model, src.loss, src.drift


__all__ = ["Evaluer"]


class Evaluer:
    def __init__(
        self, data_save_fold: str | None, ckpt_load_path: str, 
        temp_save_fold: str | None,
        stride: int | None, window: int |  None, method: str | None, 
        batch_size: int, num_workers: int,
        evaluset: src.data.RawDataset, 
    ) -> None:
        # evalu
        self.device = "cuda"
        # path
        self.data_save_fold = os.path.normpath(
            data_save_fold
        ) if data_save_fold else None
        self.ckpt_load_path = os.path.normpath(ckpt_load_path)
        self.temp_save_fold = os.path.normpath(
            temp_save_fold
        ) if temp_save_fold else None
        # drift
        self.stride = stride
        self.window = window
        self.method = method    # None, "DCC", "MCC", or "RCC"

        # data
        self.evaluset = evaluset
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.evaluset,
            batch_size=batch_size, num_workers=num_workers, 
            pin_memory=True, persistent_workers=True
        )
        # model
        ckpt = torch.load(self.ckpt_load_path, map_location=self.device)
        self.model = src.model.ResAttUNet(
            ckpt["dim"], ckpt["feats"], ckpt["use_cbam"], ckpt["use_res"]
        ).to(self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.half()
        self.model.eval()

        # index
        # number of subframes of each frame
        self.num_sub_user = torch.prod(self.evaluset.num_sub_user)
        # number of subframes of each batch of prediction
        self.batch_size = self.dataloader.batch_size
        if self.num_sub_user % self.batch_size != 0: raise ValueError(
            "num_sub_user_prod must divisible by batch_size, but got {} and {}"
            .format(self.num_sub_user, self.batch_size)
        )
        self.total = len(self.dataloader)//(self.num_sub_user/self.batch_size)

        # print model info
        """
        para_num = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f'The model has {para_num:,} trainable parameters')
        """

    @torch.no_grad()
    def fit(self) -> None:
        # case 1: if self.method is not given, do not perform drift correction
        if self.method == None:
            sub_cat = torch.zeros(
                self.num_sub_user, *self.evaluset.dim_dst_pad, 
                dtype=torch.float16, device=self.device
            )
            for i, frames in tqdm.tqdm(
                enumerate(self.dataloader), 
                desc=self.data_save_fold, total = len(self.dataloader), 
                unit="frame", dynamic_ncols=True, smoothing=0.0,
                unit_scale=float(1/(self.num_sub_user/self.batch_size)),
            ):
                frame_index = int(i // (self.num_sub_user/self.batch_size))
                sub_index   = int(i  % (self.num_sub_user/self.batch_size))
                # prediction of batch_size patches of the current frame
                sub_cat[
                    sub_index*self.batch_size : (sub_index+1)*self.batch_size,
                    :, :, :
                ] += self.model(frames.half().to(self.device))
                # continue if we haven't complete the prediction of all patches 
                # of the current frame
                if (sub_index+1)*self.batch_size < self.num_sub_user: continue
                # save the current result as .tif
                if (frame_index+1) & frame_index and \
                frame_index+1 != self.total and \
                (frame_index+1) % 1000 != 0: continue
                self._save(sub_cat, frame_index, self.data_save_fold)
        # case 2: perform drift correction; first we predict and save temp 
        # result for drift correction; then, perform drift correction using the 
        # temp result in self.temp_save_fold saved
        elif self.stride != 0 or self.window != 0:
            # predict and save temp result for drift correction
            if os.path.exists(self.temp_save_fold) and \
            len(os.listdir(self.temp_save_fold)) >= 2:
                # before predict, check if self.temp_save_fold exists; 
                # if self.temp_save_fold exists, calculate the stride in 
                # self.temp_save_fold and compare with self.stride.
                idx = [
                    int(file.split('.')[0]) 
                    for file in os.listdir(self.temp_save_fold)
                    if file.endswith('.tif')
                ]
                idx.sort()
                # if the stride in temp_save_fold is the same as given stride,
                # skip the prediction and directly perform drift correction
                if idx[1]-idx[0] == self.stride: print(
                    "The stride in temp_save_fold " + 
                    "`{}` ".format(self.temp_save_fold) + 
                    "same as given stride. Skip the prediction."
                )
                # if the stride in temp_save_fold is different from given 
                # stride, means we have a ambiguous situation and raise an error
                if idx[1]-idx[0] != self.stride: raise ValueError(
                    "The stride in temp_save_fold " + 
                    "`{}` ".format(self.temp_save_fold) + 
                    "diff from given stride. Please either delete the " + 
                    "`{}` ".format(self.temp_save_fold) + 
                    "to use the given stride or change the " + 
                    "given stride to the stride in " + 
                    "`{}`.".format(self.temp_save_fold)
                )
            else:
                # if self.temp_save_fold does not exist or is empty, perform
                # prediction and save temp result for drift correction
                sub_cat = torch.zeros(
                    self.num_sub_user, *self.evaluset.dim_dst_pad, 
                    dtype=torch.float16, device=self.device
                )
                for i, frames in tqdm.tqdm(
                    enumerate(self.dataloader), 
                    desc=self.temp_save_fold, total = len(self.dataloader), 
                    unit="frame", dynamic_ncols=True, smoothing=0.0,
                    unit_scale=float(1/(self.num_sub_user/self.batch_size)),
                ):
                    frame_index = int(i // (self.num_sub_user/self.batch_size))
                    sub_index   = int(i  % (self.num_sub_user/self.batch_size))
                    # prediction of batch_size patches of the current frame
                    sub_cat[
                        sub_index*self.batch_size : 
                        (sub_index+1)*self.batch_size,
                        :, :, :
                    ] += self.model(frames.half().to(self.device))
                    # continue if we haven't complete the prediction of all  
                    # patches of the current frame
                    if (sub_index+1)*self.batch_size < self.num_sub_user:
                        continue
                    # save the current result as .tif
                    if self.stride == 0: continue
                    if (frame_index+1) % self.stride != 0: continue
                    self._save(sub_cat, frame_index, self.temp_save_fold)
                    sub_cat = torch.zeros(
                        self.num_sub_user, *self.evaluset.dim_dst_pad, 
                        dtype=torch.float16, device=self.device
                    )

            # perform drift correction using the temp result in 
            # self.temp_save_fold
            src.drift.DriftCorrector(
                self.temp_save_fold, self.window, self.method
            ).fit()

            # we need to exit the program since temp result for drift correction
            # and final prediction may use different region of the frame, i.e., 
            # a small region for getting temp result to reduce the time of 
            # calculating the drift and a large region for the final prediction.
            return
        # case 3:
        # we already have cached {self.method}.csv, means we want to perform the 
        # final prediction
        else:
            cache_path = os.path.join(
                self.temp_save_fold, "{}.csv".format(self.method)
            )
            drift = torch.from_numpy(np.loadtxt(cache_path, delimiter=','))
            print(
                "Load drift from `{}`. ".format(cache_path) + 
                "Delete `{}` if you want to re-calculate ".format(cache_path) + 
                "the drift for same dataset with new window size. " + 
                "Delete whole `{}` if you want ".format(self.temp_save_fold) + 
                "to re-calculate the drift for same dataset with new stride " + 
                "size. " + 
                "Delete whole `{}` or specify a ".format(self.temp_save_fold) + 
                "new path (recommend) if you want to re-calculate the drift " + 
                "for different dataset."
            )
            # scale up the drift to increase the percision since for image, we
            # can only shift the image by integer pixels
            drift *= self.evaluset.scale
            drift  = drift.round().int()

            sub_cat = torch.zeros(
                self.num_sub_user, *self.evaluset.dim_dst_pad*4, 
                dtype=torch.float16, device=self.device
            )
            for i, frames in tqdm.tqdm(
                enumerate(self.dataloader), 
                desc=self.data_save_fold, total = len(self.dataloader), 
                unit="frame", dynamic_ncols=True, smoothing=0.0,
                unit_scale=float(1/(self.num_sub_user/self.batch_size)),
            ):
                frame_index = int(i // (self.num_sub_user/self.batch_size))
                sub_index   = int(i  % (self.num_sub_user/self.batch_size))
                # prediction of batch_size patches of the current frame
                result = self.model(frames.half().to(self.device))
                for i in range(len(result)):
                    sub = F.interpolate(
                        result[i].unsqueeze(0).unsqueeze(0), 
                        scale_factor=(4, 4, 4)
                    ).squeeze(0).squeeze(0)
                    sub_cat[sub_index*self.batch_size+i] += torch.roll(
                        sub, tuple(drift[frame_index]), dims=(0, 1, 2)
                    )
                # continue if we haven't complete the prediction of all patches 
                # of the current frame
                if (sub_index+1)*self.batch_size < self.num_sub_user: continue
                # save the current result as .tif
                if (frame_index+1) & frame_index and \
                frame_index+1 != self.total and \
                (frame_index+1) % 1000 != 0: continue
                self._save(
                        F.interpolate(
                        sub_cat.unsqueeze(0), 
                        scale_factor=(1/4, 1/4, 1/4)
                    ).squeeze(0), frame_index, self.data_save_fold
                )

    @torch.no_grad()
    def _save(self, sub_cat: Tensor, frame_index: int, fold: str) -> None:
        """
        Save the result of the current frame as .tif.

        Args:
            sub_cat (Tensor): The prediction of the current frame.
            frame_index (int): The index of the current frame, start from 0.
            fold (str): The folder to save the result.
        """
        # check if the folder exists every time before saving
        if not os.path.exists(fold): os.makedirs(fold)

        # save as float32 tif
        tifffile.imwrite(
            os.path.join(fold, "{:05}.tif".format(frame_index+1)),
            self.evaluset.combineFrame(
                sub_cat
            ).detach().cpu().float().numpy(),
        )
