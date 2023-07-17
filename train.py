import torch
import torch.cuda.amp as amp
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.tensorboard.writer as writer

import os
import tqdm

import config, data, model


torch.backends.cudnn.enabled = True     # type: ignore
torch.backends.cudnn.benchmark = True   # type: ignore


class Train:
    def __init__(self, config) -> None:
        # train
        self.device = config.device
        self.max_epoch = config.max_epoch
        self.accumu_steps = config.accumu_steps
        # learning rate
        self.lr    = config.lr
        self.gamma = config.gamma
        # checkpoint
        self.ckpt_save_folder = config.ckpt_save_folder
        self.ckpt_load_path   = config.ckpt_load_path
        self.ckpt_load_lr     = config.ckpt_load_lr

        # index
        self.epoch = 1  # epoch index may update in load_ckpt()

        # data
        self.trainloader, self.validloader = data.getDataLoader(config)
        # model
        self.net       = model.ResUNet2D(config).to(self.device)
        self.criterion = model.Criterion(config).to(self.device)

        # optimizer
        self.scaler    = amp.GradScaler()  # type: ignore
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.gamma)
        
        # record
        self.writer = writer.SummaryWriter()

    def train(self) -> None:
        self.load_ckpt()
        for self.epoch in tqdm.tqdm(
            range(self.epoch, self.max_epoch+1), 
            total=self.max_epoch, desc=self.ckpt_save_folder, smoothing=0.0,
            unit="epoch", initial=self.epoch
        ):
            self.train_epoch()
            self.valid_epoch()
            self.update_lr()
            self.save_ckpt()

    def train_epoch(self) -> None:
        self.net.train()

        # record: progress bar
        pbar = tqdm.tqdm(
            total=int(len(self.trainloader)/self.accumu_steps), 
            desc='train_epoch', leave=False, unit="steps", smoothing=1.0
        )

        for i, (frames, labels) in enumerate(self.trainloader):
            # put frames and labels in GPU
            frames = frames.to(self.device)
            labels = labels.to(self.device)

            # forward and backward
            with amp.autocast(dtype=torch.float16):  # type: ignore
                outputs = self.net(frames)
                loss = self.criterion(outputs, labels) / self.accumu_steps
            self.scaler.scale(loss).backward()  # type: ignore

            # update model parameters
            if (i+1) % self.accumu_steps != 0: continue
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # record: tensorboard
            self.writer.add_scalars(
                'Loss', {'train': loss.item() / len(outputs)}, 
                (self.epoch - 1) * len(self.trainloader) / self.accumu_steps + 
                (i + 1) / self.accumu_steps
            )
            self.writer.add_scalars(
                'Num', {'train': len(torch.nonzero(outputs)) / len(outputs)}, 
                (self.epoch - 1) * len(self.trainloader) / self.accumu_steps + 
                (i + 1) / self.accumu_steps
            )
            # record: progress bar
            pbar.update()

    @torch.no_grad()
    def valid_epoch(self) -> None:
        self.net.eval()

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
            outputs = self.net(frames)
            # loss
            loss = self.criterion(outputs, labels) / self.accumu_steps

            # record: tensorboard
            valid_loss.append(loss.item() / len(outputs))
            valid_num.append(len(torch.nonzero(outputs)) / len(outputs))
            # record: progress bar
            if (i+1) % self.accumu_steps == 0: pbar.update()
        
        # record: tensorboard
        self.writer.add_scalars(
            'Loss', {'valid': torch.mean(torch.as_tensor(valid_loss))}, 
            self.epoch * len(self.trainloader) / self.accumu_steps
        )
        self.writer.add_scalars(
            'Num', {'valid': torch.mean(torch.as_tensor(valid_num))}, 
            self.epoch * len(self.trainloader) / self.accumu_steps
        )

    @torch.no_grad()
    def update_lr(self) -> None:
        # update learning rate
        if self.scheduler.get_last_lr()[0] > 1e-8: self.scheduler.step()

        # record: tensorboard
        self.writer.add_scalar(
            'LR', self.optimizer.param_groups[0]['lr'], 
            self.epoch * len(self.trainloader) / self.accumu_steps
        )

    @torch.no_grad()
    def save_ckpt(self) -> None:
        # file path checking
        if not os.path.exists(self.ckpt_save_folder): 
            os.makedirs(self.ckpt_save_folder)

        torch.save({
            'epoch': self.epoch,  # epoch index start from 1
            'net': self.net.state_dict(),
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
            }, "{}/{}.ckpt".format(self.ckpt_save_folder, self.epoch))

    @torch.no_grad()
    def load_ckpt(self) -> None:
        if self.ckpt_load_path == "": return
        ckpt = torch.load("{}.ckpt".format(self.ckpt_load_path))
        self.epoch = ckpt['epoch']+1  # start train from next epoch index
        self.net.load_state_dict(ckpt['net'])
        self.scaler.load_state_dict(ckpt['scaler'])
        if self.ckpt_load_lr:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])


if __name__ == "__main__":
    trainer = Train(config.getConfig("train"))
    trainer.train()
