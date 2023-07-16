import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.tensorboard.writer as writer

import os
import tqdm

import config, data, model


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
        self.net       = model.UNet2D(config).to(self.device)
        self.criterion = model.Criterion(config).to(self.device)

        # optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.gamma)
        
        # record
        self.writer = writer.SummaryWriter()

    def train(self) -> None:
        self.load_ckpt()
        for self.epoch in tqdm.tqdm(
            range(self.epoch, self.max_epoch+1), 
            total=self.max_epoch, desc=self.ckpt_save_folder, position=0,
            unit="epoch", initial=self.epoch
        ):
            self.train_epoch()
            self.valid_epoch()
            self.update_lr()
            self.save_ckpt()

    def train_epoch(self) -> None:
        self.net.train()
        for i, (frames, labels) in enumerate(tqdm.tqdm(
            self.trainloader, desc='train_epoch', position=1, 
            leave=False, unit="iteration"
        )):
            # put frames and labels in GPU
            frames = frames.to(self.device)
            labels = labels.to(self.device)

            # forward and backward
            outputs = self.net(frames)
            loss = self.criterion(outputs, labels) / self.accumu_steps
            loss.backward()

            # update model parameters
            if (i+1) % self.accumu_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # record
            self.writer.add_scalars(
                'Loss', {'train': loss.item() / len(outputs)}, 
                (self.epoch-1)*len(self.trainloader)+i)
            self.writer.add_scalars(
                'Num', {'train': len(torch.nonzero(outputs)) / len(outputs)}, 
                (self.epoch-1)*len(self.trainloader)+i)
    
    @torch.no_grad()
    def valid_epoch(self) -> None:
        valid_loss = []
        valid_num = []
        
        self.net.eval()
        for _, (frames, labels) in enumerate(tqdm.tqdm(
            self.validloader, desc='valid_epoch', position=1, 
            leave=False, unit="iteration"
        )):
            # put frames and labels in GPU
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            # forward
            outputs = self.net(frames)
            # loss
            loss = self.criterion(outputs, labels) / self.accumu_steps

            # record
            valid_loss.append(loss.item() / len(outputs))
            valid_num.append(len(torch.nonzero(outputs)) / len(outputs))
        
        # validation loss
        valid_loss = torch.mean(torch.as_tensor(valid_loss))
        valid_num = torch.mean(torch.as_tensor(valid_num))
        
        # record
        self.writer.add_scalars(
            'Loss', {'valid': valid_loss}, 
            self.epoch*len(self.trainloader))
        self.writer.add_scalars(
            'Num', {'valid': valid_num}, 
            self.epoch*len(self.trainloader))

    @torch.no_grad()
    def update_lr(self) -> None:
        # update learning rate
        if self.scheduler.get_last_lr()[0] > 1e-8: self.scheduler.step()

        # record
        self.writer.add_scalar(
            'LR', self.optimizer.param_groups[0]['lr'], 
            self.epoch*len(self.trainloader))

    @torch.no_grad()
    def save_ckpt(self) -> None:
        # file path checking
        if not os.path.exists(self.ckpt_save_folder): 
            os.makedirs(self.ckpt_save_folder)

        torch.save({
            'epoch': self.epoch,  # epoch index start from 1
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
            }, "{}/{}.ckpt".format(self.ckpt_save_folder, self.epoch))

    @torch.no_grad()
    def load_ckpt(self) -> None:
        if self.ckpt_load_path == "": return
        ckpt = torch.load("{}.ckpt".format(self.ckpt_load_path))
        self.epoch = ckpt['epoch']+1  # start train from next epoch index
        self.net.load_state_dict(ckpt['net'])
        if self.ckpt_load_lr:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])


if __name__ == "__main__":
    trainer = Train(config.getConfig("train"))
    trainer.train()
