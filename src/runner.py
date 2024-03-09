import torch
import torch.utils.data                 # DataLoader
import torch.cuda.amp                   # GradScaler, autocast
import torch.optim                      # Adam
import torch.optim.lr_scheduler         # ExponentialLR
import torch.utils.tensorboard.writer   # SummaryWriter
import torch.nn.functional as F
from torch import Tensor

import numpy as np

import os
import sys
import tifffile
if 'ipykernel' in sys.modules:
    import tqdm.notebook as tqdm
else:
    import tqdm     # since tqdm does not work in jupyter properly

import src.data, src.model, src.loss, src.drift

__all__ = ["Evaluer", "Trainer"]


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
        ckpt = torch.load(
            "{}.ckpt".format(self.ckpt_load_path), map_location=self.device
        )
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
                enumerate(self.dataloader), desc=self.data_save_fold,
                total = len(self.dataloader), unit="frame", 
                unit_scale=float(1/(self.num_sub_user/self.batch_size)),
                dynamic_ncols=True, smoothing=0.0,
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
                frame_index+1 != self.total: continue
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
                    enumerate(self.dataloader), desc=self.temp_save_fold,
                    total = len(self.dataloader), unit="frame", 
                    unit_scale=float(1/(self.num_sub_user/self.batch_size)),
                    dynamic_ncols=True, smoothing=0.0,
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
                enumerate(self.dataloader), desc=self.data_save_fold,
                total = len(self.dataloader), unit="frame", 
                unit_scale=float(1/(self.num_sub_user/self.batch_size)),
                dynamic_ncols=True, smoothing=0.0,
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
                frame_index+1 != self.total: continue
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


class Trainer:
    def __init__(
        self, max_epoch: int, accumu_steps: int, 
        ckpt_save_fold: str, ckpt_load_path: str, ckpt_load_lr: bool,
        batch_size: int, lr: float,
        trainset: src.data.SimDataset, validset: src.data.RawDataset, 
        model: src.model.ResAttUNet,
    ) -> None:
        # train
        self.device = "cuda"
        self.max_epoch = max_epoch
        self.accumu_steps = accumu_steps
        # checkpoint
        self.ckpt_save_fold = ckpt_save_fold
        self.ckpt_load_path = ckpt_load_path
        self.ckpt_load_lr   = ckpt_load_lr

        # data
        self.trainloader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=batch_size, num_workers=batch_size*4, pin_memory=True
        )
        self.validloader = torch.utils.data.DataLoader(
            dataset=validset, 
            batch_size=batch_size, num_workers=batch_size*4, pin_memory=True
        )
        # model
        self.model = model.to(self.device)
        # loss
        self.loss  = src.loss.GaussianBlurLoss().to(self.device)
        # optimizer
        self.scaler    = torch.cuda.amp.GradScaler()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.95
        )
        # recorder
        self.writer = torch.utils.tensorboard.writer.SummaryWriter()

        # index
        self.epoch = 1  # epoch index may update in load_ckpt()

        # print model info
        """
        para_num = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f'The model has {para_num:,} trainable parameters')
        """

    def fit(self) -> None:
        self._loadCkpt()
        for self.epoch in tqdm.tqdm(
            range(self.epoch, self.max_epoch+1), 
            total=self.max_epoch, desc=self.ckpt_save_fold, smoothing=0.0,
            unit="epoch", initial=self.epoch, dynamic_ncols=True,
        ):
            self._trainEpoch()
            self._validEpoch()
            self._updateLr()
            self._saveCkpt()

    def _trainEpoch(self) -> None:
        self.model.train()

        # record: progress bar
        pbar = tqdm.tqdm(
            total=int(len(self.trainloader)/self.accumu_steps), 
            desc='_trainEpoch', leave=False, unit="steps", smoothing=1.0, 
            dynamic_ncols=True,
        )
        # record: tensorboard
        train_loss = []
        train_num  = []

        for i, (frames, labels) in enumerate(self.trainloader):
            # put frames and labels in GPU
            frames = frames.to(self.device)
            labels = labels.to(self.device)

            # forward and backward
            with torch.cuda.amp.autocast(dtype=torch.float16):
                predis = self.model(frames)
                loss_value = self.loss(predis, labels) / self.accumu_steps
            self.scaler.scale(loss_value).backward()

            # record: tensorboard
            train_loss.append(loss_value.item() / len(predis))
            train_num.append(len(torch.nonzero(predis)) / len(predis))

            # update model parameters
            if (i+1) % self.accumu_steps != 0: continue
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # record: tensorboard
            self.writer.add_scalars(
                'scalars/loss', 
                {'train': torch.sum(torch.as_tensor(train_loss))}, 
                (self.epoch-1) * len(self.trainloader) / self.accumu_steps + 
                (i+1) / self.accumu_steps
            )  # average loss of each frame
            self.writer.add_scalars(
                'scalars/numb', 
                {'train': torch.mean(torch.as_tensor(train_num))}, 
                (self.epoch-1) * len(self.trainloader) / self.accumu_steps + 
                (i+1) / self.accumu_steps
            )  # average num of each frame
            train_loss = []
            train_num  = []
            # record: progress bar
            pbar.update()

    @torch.no_grad()
    def _validEpoch(self) -> None:
        self.model.eval()

        # record: progress bar
        pbar = tqdm.tqdm(
            total=int(len(self.validloader)/self.accumu_steps), 
            desc="_validEpoch", leave=False, unit="steps", smoothing=1.0, 
            dynamic_ncols=True,
        )
        # record: tensorboard
        valid_loss = []
        valid_num  = []

        for i, (frames, labels) in enumerate(self.validloader):
            # put frames and labels in GPU
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            # forward
            predis = self.model(frames)
            # loss
            loss_value = self.loss(predis, labels)

            # record: tensorboard
            valid_loss.append(loss_value.item() / len(predis))
            valid_num.append(len(torch.nonzero(predis)) / len(predis))
            # record: progress bar
            if (i+1) % self.accumu_steps == 0: pbar.update()
        
        # record: tensorboard
        self.writer.add_scalars(
            'scalars/loss', {'valid': torch.mean(torch.as_tensor(valid_loss))}, 
            self.epoch * len(self.trainloader) / self.accumu_steps
        )
        self.writer.add_scalars(
            'scalars/numb', {'valid': torch.mean(torch.as_tensor(valid_num))}, 
            self.epoch * len(self.trainloader) / self.accumu_steps
        )

    @torch.no_grad()
    def _updateLr(self) -> None:
        # update learning rate
        if self.scheduler.get_last_lr()[0] > 1e-10: self.scheduler.step()

        # record: tensorboard
        self.writer.add_scalar(
            'scalars/lr', self.optimizer.param_groups[0]['lr'], 
            self.epoch * len(self.trainloader) / self.accumu_steps
        )

    @torch.no_grad()
    def _saveCkpt(self) -> None:
        # file path checking
        if not os.path.exists(self.ckpt_save_fold): 
            os.makedirs(self.ckpt_save_fold)

        torch.save({
            'epoch': self.epoch,  # epoch index start from 1
            'model': self.model.state_dict(),
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            "dim": self.model.dim,
            "feats": self.model.feats,
            "use_cbam": self.model.use_cbam,
            "use_res": self.model.use_res,
        }, "{}/{}.ckpt".format(self.ckpt_save_fold, self.epoch))

    @torch.no_grad()
    def _loadCkpt(self) -> None:
        if self.ckpt_load_path == "": return
        ckpt = torch.load("{}.ckpt".format(self.ckpt_load_path))
        
        self.epoch = ckpt['epoch']+1  # start train from next epoch index
        self.model.load_state_dict(ckpt['model'], strict=False)
        self.scaler.load_state_dict(ckpt['scaler'])
        
        if not self.ckpt_load_lr: return
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
