import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.tensorboard.writer as writer
from torch.utils.data import DataLoader

import os
import tifffile

import tqdm

import sml.data, sml.model, sml.loss

__all__ = ["Trainer", "Evaluer"]


class Trainer:
    def __init__(
        self, max_epoch: int, accumu_steps: int, 
        ckpt_save_fold: str, ckpt_load_path: str, ckpt_load_lr: bool,
        trainset: sml.data.SimDataset, validset: sml.data.RawDataset, 
        batch_size: int, num_workers: int,
        model: sml.model.ResAttUNet, lr: float, gamma: float,
    ) -> None:
        self.device = "cuda"
        self.max_epoch = max_epoch
        self.accumu_steps = accumu_steps
        # path
        self.ckpt_save_fold = ckpt_save_fold
        self.ckpt_load_path = ckpt_load_path
        self.ckpt_load_lr   = ckpt_load_lr

        # data
        self.trainloader = DataLoader(
            trainset,
            batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )
        self.validloader = DataLoader(
            validset, 
            batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )
        # model
        self.model = model.to(self.device)
        # loss
        self.loss  = sml.loss.GaussianBlurLoss().to(self.device)
        # optimizer
        self.scaler    = amp.GradScaler()  # type: ignore
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ExponentialLR(
            self.optimizer, gamma=gamma
        )
        # recorder
        self.writer = writer.SummaryWriter()

        # index
        self.epoch = 1  # epoch index may update in load_ckpt()

        # print model info
        para_num = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f'The model has {para_num:,} trainable parameters')

    def fit(self) -> None:
        self._load_ckpt()
        for self.epoch in tqdm.tqdm(
            range(self.epoch, self.max_epoch+1), 
            total=self.max_epoch, desc=self.ckpt_save_fold, smoothing=0.0,
            unit="epoch", initial=self.epoch
        ):
            self._train_epoch()
            self._valid_epoch()
            self._update_lr()
            self._save_ckpt()

    def _train_epoch(self) -> None:
        self.model.train()

        # record: progress bar
        pbar = tqdm.tqdm(
            total=int(len(self.trainloader)/self.accumu_steps), 
            desc='train_epoch', leave=False, unit="steps", smoothing=1.0
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
    def _valid_epoch(self) -> None:
        self.model.eval()

        # record: progress bar
        pbar = tqdm.tqdm(
            total=int(len(self.validloader)/self.accumu_steps), 
            desc="valid_epoch", leave=False, unit="steps", smoothing=1.0
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
    def _update_lr(self) -> None:
        # update learning rate
        if self.scheduler.get_last_lr()[0] > 1e-10: self.scheduler.step()

        # record: tensorboard
        self.writer.add_scalar(
            'scalars/lr', self.optimizer.param_groups[0]['lr'], 
            self.epoch * len(self.trainloader) / self.accumu_steps
        )

    @torch.no_grad()
    def _save_ckpt(self) -> None:
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
    def _load_ckpt(self) -> None:
        if self.ckpt_load_path == "": return
        ckpt = torch.load("{}.ckpt".format(self.ckpt_load_path))
        
        self.epoch = ckpt['epoch']+1  # start train from next epoch index
        self.model.load_state_dict(ckpt['model'], strict=False)
        self.scaler.load_state_dict(ckpt['scaler'])
        
        if not self.ckpt_load_lr: return
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])


class Evaluer:
    def __init__(
        self, segpara: int, data_save_fold: str, ckpt_load_path: str,
        evaluset: sml.data.RawDataset, batch_size: int, num_workers: int,
    ) -> None:
        self.device = "cuda"
        self.segpara = segpara
        # path
        self.data_save_fold = data_save_fold
        self.ckpt_load_path = ckpt_load_path

        # data
        self.evaluset = evaluset
        self.dataloader = DataLoader(
            self.evaluset,
            batch_size=batch_size, num_workers=num_workers, pin_memory=True
        )
        # model
        ckpt = torch.load(
            "{}.ckpt".format(self.ckpt_load_path), 
            map_location=self.device
        )
        self.model = sml.model.ResAttUNet(
            ckpt["dim"], ckpt["feats"], ckpt["use_cbam"], ckpt["use_res"]
        ).to(self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.half()

        # dim
        self.dim_dst_pad = self.evaluset.dim_dst_pad     # [D], int
        self.dim_dst_raw = self.evaluset.dim_dst_raw     # [D], int
        # index
        self.num_sub_user = self.evaluset.num_sub_user   # [D], int
        self.num_sub_user_prod = torch.prod(self.num_sub_user)
        self.batch_size = self.dataloader.batch_size
        if self.num_sub_user_prod % self.batch_size != 0: raise ValueError(
            "num_sub_user_prod must divisible by batch_size, but got {} and {}"
            .format(self.num_sub_user_prod, self.batch_size)
        )

        # print model info
        para_num = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f'The model has {para_num:,} trainable parameters')

    @torch.no_grad()
    def fit(self) -> None:
        self.model.eval()

        # record: progress bar
        pbar = tqdm.tqdm(
            total=int(
                len(self.dataloader)*self.batch_size/self.num_sub_user_prod
            ), desc=self.data_save_fold, unit="frame"
        )

        sub_cat = torch.zeros(
            self.num_sub_user_prod, *self.dim_dst_pad, 
            dtype=torch.float16, device=self.device
        )
        previous = None
        for i, frames in enumerate(self.dataloader):
            frame_index = int(i // (self.num_sub_user_prod/self.batch_size))
            sub_index   = int(i  % (self.num_sub_user_prod/self.batch_size))

            # prediction of batch_size patches of the current frame
            sub_cat[
                sub_index * self.batch_size : (sub_index+1) * self.batch_size,
                :, :, :
            ] += self.model(frames.half().to(self.device))

            # continue if we haven't complete the prediction of all patches of
            # the current frame
            if (sub_index+1)*self.batch_size < self.num_sub_user_prod: continue

            # save after combine 1, 2, 4, 8, 16... frames
            if not (frame_index+1) & frame_index or i == len(self.dataloader)-1:
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

            # TODO: Note that this is a temporary code for testing drift 
            # correction. We will change to real-time drift correction later

            # save after combine segpara num of frames and then reset
            if self.segpara != 0 and (frame_index+1) % self.segpara == 0:
                # check if the folder exists every time before saving
                if not os.path.exists(os.path.join(self.data_save_fold, "seg")): 
                    os.makedirs(os.path.join(self.data_save_fold, "seg"))
                # initialize previous as zeros if it's None
                if previous is None:
                    previous = torch.zeros_like(
                        self.evaluset.combineFrame(sub_cat)
                    ).detach().cpu().numpy()
                # save as float16 tif
                tifffile.imwrite(
                    os.path.join(
                        self.data_save_fold, "seg",
                        "{:05}.tif".format(frame_index+1)
                    ),
                    self.evaluset.combineFrame(
                        sub_cat
                    ).detach().cpu().numpy() - previous,
                )
                # save current frame for next round
                previous = self.evaluset.combineFrame(
                    sub_cat
                ).detach().cpu().numpy()

            pbar.update()  # update progress bar
