import torch
import torch.cuda.amp as amp
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.tensorboard.writer as writer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import Tensor

import numpy as np

import os
import tifffile
import tqdm
import warnings

import sml.data, sml.model, sml.loss, sml.drift

__all__ = ["Evaluer", "Trainer"]


class Evaluer:
    def __init__(
        self, data_save_fold: str, ckpt_load_path: str, temp_save_fold: str,
        stride: int, window: int, batch_size: int,
        evaluset: sml.data.RawDataset, 
    ) -> None:
        # evalu
        self.device = "cuda"
        # path
        self.data_save_fold = data_save_fold
        self.ckpt_load_path = ckpt_load_path
        self.temp_save_fold = temp_save_fold
        # drift
        self.stride = stride
        self.window = window

        # data
        self.evaluset = evaluset
        self.dataloader = DataLoader(
            dataset=self.evaluset,
            batch_size=batch_size, num_workers=batch_size*4, pin_memory=True
        )
        # model
        ckpt = torch.load(
            "{}.ckpt".format(self.ckpt_load_path), map_location=self.device
        )
        self.model = sml.model.ResAttUNet(
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
        if self.stride != 0 and self.window != 0: 
            raise ValueError(
                "Stride and window cannot be set as non-zero at the same " + 
                "time. We split the drifting correction into two steps: " + 
                "save the result for drifting correction by set stride and " + 
                "perform the drifting correction by set window. " + 
                "However, they may use different region of the frame, i.e., " + 
                "we may use a small region for first step to reduce the time " +
                "of calculating the drift and a large region for the final " + 
                "prediction. Thus, please re-run the code and either set " + 
                "stride or window as 0." 
            )

        # when stride is not 0 and window is 0, means we need to save result for
        # drift correction by calling self._saveStride and do not need to 
        # perform the drift correction.
        if self.stride != 0 and self.window == 0:
            # before evaluation, check if self.temp_save_fold exists. 
            # If self.temp_save_fold exists, calculate the stride in 
            # self.temp_save_fold and compare with self.stride.
            if os.path.exists(self.temp_save_fold) and \
            len(os.listdir(self.temp_save_fold)) >= 2:
                idx = [
                    int(file.split('.')[0]) 
                    for file in os.listdir(self.temp_save_fold)
                ]
                idx.sort()
                if idx[1]-idx[0] == self.stride: warnings.warn(
                    "The stride in temp_save_fold " + 
                    "`{}` ".format(self.temp_save_fold) + 
                    "is the same as given stride. Skip the evaluation.",
                    UserWarning
                )
                if idx[1]-idx[0] != self.stride: raise ValueError(
                    "The stride in temp_save_fold " + 
                    "`{}` ".format(self.temp_save_fold) + 
                    "is diff from given stride. Please either delete the " + 
                    "temp_save_fold to use the given stride or change the " + 
                    "given stride to the stride in temp_save_fold."
                )
                return

            sub_cat = torch.zeros(
                self.num_sub_user, *self.evaluset.dim_dst_pad, 
                dtype=torch.float16, device=self.device
            )
            for i, frames in tqdm.tqdm(
                enumerate(self.dataloader), desc=self.temp_save_fold,
                total = len(self.dataloader), unit="frame", 
                unit_scale=float(1/(self.num_sub_user/self.batch_size)),
                dynamic_ncols=True,
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
                if self.stride == 0: continue
                if (frame_index+1) % self.stride != 0: continue
                self._saveStride(sub_cat, frame_index)

        # when stride is 0 and window is not 0, means we already save the result
        # for drift correction in self.temp_save_fold by setting stride not 
        # equal to 0 before, and now we need to perform the drift correction. 
        # We will save the result before and after drift correction for 
        # comparison.
        if self.stride == 0 and self.window != 0:
            drift = self._getDrift()
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
                dynamic_ncols=True,
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
                self._saveAccumu(F.interpolate(
                    sub_cat.unsqueeze(0), 
                    scale_factor=(1/4, 1/4, 1/4)
                ).squeeze(0), frame_index)

        # when both stride and window are 0, we don't need to perform drift
        # correction; just predict and save the result directly. 
        if self.stride == 0 and self.window == 0:
            sub_cat = torch.zeros(
                self.num_sub_user, *self.evaluset.dim_dst_pad, 
                dtype=torch.float16, device=self.device
            )
            for i, frames in tqdm.tqdm(
                enumerate(self.dataloader), desc=self.data_save_fold,
                total = len(self.dataloader), unit="frame", 
                unit_scale=float(1/(self.num_sub_user/self.batch_size)),
                dynamic_ncols=True,
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
                self._saveAccumu(sub_cat, frame_index)

    @torch.no_grad()
    def _getDrift(self) -> Tensor:
        # error for no result saving in self.temp_save_fold for drift correction
        error = ValueError(
            "No result saving in temp_save_fold " +
            "`{}` for drifting correction. ".format(self.temp_save_fold) +
            "Please re-run the code and set stride not equal to 0."
        )

        drift = None
        # if self.temp_save_fold not exists, we must have not save the result
        if not os.path.exists(self.temp_save_fold):
            raise error
        # find the drift.csv in self.temp_save_fold, directly load cache
        elif os.path.exists(os.path.join(self.temp_save_fold, "drift.csv")):
            drift = np.loadtxt(
                os.path.join(self.temp_save_fold, "drift.csv"), 
                delimiter=','
            )
            warnings.warn(
                "Load drift from `{}`. ".format(
                    os.path.join(self.temp_save_fold, "drift.csv")
                ) + "Please delete the .csv file if you want to " + 
                "recalculate the drift instead of load from cache.",
                UserWarning
            )
        # cache for drift correction found, calculate the drift and save as .csv
        else:
            # get list of the stride of the self.temp_save_fold
            idx = [
                int(file.split('.')[0]) 
                for file in os.listdir(self.temp_save_fold)
                if file.endswith('.tif')
            ]
            idx.sort()
            # if no .tif file found, raise error
            if not idx: raise error
            # calculate the drift
            drift = sml.drift.DriftCorrector(
                total=idx[-1], stride=idx[1]-idx[0], window=self.window,
                load_fold=self.temp_save_fold
            ).fit()
            # save the drift as .csv
            np.savetxt(
                os.path.join(self.temp_save_fold, "drift.csv"), 
                drift, delimiter=','
            )
        return torch.from_numpy(drift)

    @torch.no_grad()
    def _saveAccumu(self, sub_cat: Tensor, frame_index: int) -> None:
        """
        Save the result of the current frame as .tif.

        Args:
            frame_index (int): The index of the current frame, start from 0.
        """
        # check if the folder exists every time before saving
        if not os.path.exists(self.data_save_fold): 
            os.makedirs(self.data_save_fold)

        # save as float32 tif
        tifffile.imwrite(
            os.path.join(
                self.data_save_fold, "{:05}.tif".format(frame_index+1)
            ),
            self.evaluset.combineFrame(
                sub_cat
            ).detach().cpu().float().numpy(),
        )

    @torch.no_grad()
    def _saveStride(self, sub_cat: Tensor, frame_index: int) -> None:
        """
        Save the result of the currect frame minus the previous frame as .tif.

        We save the result as float16 since these result no need to check by
        fiji and only use to perform drift correction in the future with code.

        Args:
            frame_index (int): The index of the current frame, start from 0.
        """
        # check if the folder exists every time before saving
        if not os.path.exists(self.temp_save_fold): 
            os.makedirs(self.temp_save_fold)

        # save as float16 tif
        # the first frame, no previous frame to minus
        if frame_index+1 == self.stride:
            tifffile.imwrite(
                os.path.join(
                    self.temp_save_fold, "{:05}.tif".format(frame_index+1)
                ),
                self.evaluset.combineFrame(
                    sub_cat
                ).detach().cpu().numpy(),
            )
        # current frames minus the previous frame
        else:
            previous = tifffile.imread(os.path.join(
                self.temp_save_fold, 
                "{:05}.tif".format(frame_index+1-self.stride)
            ))
            tifffile.imwrite(
                os.path.join(
                    self.temp_save_fold, "{:05}.tif".format(frame_index+1)
                ),
                self.evaluset.combineFrame(
                    sub_cat
                ).detach().cpu().numpy() - previous,
            )


class Trainer:
    def __init__(
        self, max_epoch: int, accumu_steps: int, 
        ckpt_save_fold: str, ckpt_load_path: str, ckpt_load_lr: bool,
        batch_size: int, lr: float,
        trainset: sml.data.SimDataset, validset: sml.data.RawDataset, 
        model: sml.model.ResAttUNet,
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
        self.trainloader = DataLoader(
            dataset=trainset,
            batch_size=batch_size, num_workers=batch_size*4, pin_memory=True
        )
        self.validloader = DataLoader(
            dataset=validset, 
            batch_size=batch_size, num_workers=batch_size*4, pin_memory=True
        )
        # model
        self.model = model.to(self.device)
        # loss
        self.loss  = sml.loss.GaussianBlurLoss().to(self.device)
        # optimizer
        self.scaler    = amp.GradScaler()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        # recorder
        self.writer = writer.SummaryWriter()

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
            desc='train_epoch', leave=False, unit="steps", smoothing=1.0, 
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
            with amp.autocast(dtype=torch.float16):
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
                'scalars/loss', {'train': torch.sum(torch.as_tensor(train_loss))}, 
                (self.epoch - 1) * len(self.trainloader) / self.accumu_steps + 
                (i + 1) / self.accumu_steps
            )  # average loss of each frame
            self.writer.add_scalars(
                'scalars/numb', {'train': torch.mean(torch.as_tensor(train_num))}, 
                (self.epoch - 1) * len(self.trainloader) / self.accumu_steps + 
                (i + 1) / self.accumu_steps
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
            desc="valid_epoch", leave=False, unit="steps", smoothing=1.0, 
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
