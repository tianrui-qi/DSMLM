import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter

import os  # for file checking

from model import UNet2D, Criterion
from data import getDataLoader


class Train:
    def __init__(self, config) -> None:
        # Configurations
        # for train
        self.max_epoch  = config.max_epoch
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # for checkpoint
        self.cpt_save_path  = config.cpt_save_path
        self.cpt_save_epoch = config.cpt_save_epoch
        self.cpt_load_path  = config.cpt_load_path
        self.cpt_load_lr    = config.cpt_load_lr
        
        # index
        self.epoch      = 1    # epoch index start from 1
        self.valid_loss = torch.tensor(float("inf"))
        self.best_loss  = torch.tensor(float("inf"))
        self.valid_num  = 0

        # data
        self.trainloader, self.validloader = getDataLoader(config)
        # model
        self.net        = UNet2D(config).to(self.device)
        self.criterion  = Criterion(config).to(self.device)

        # optimizer
        self.optimizer  = optim.Adam(self.net.parameters(), lr=config.lr)
        self.scheduler  = lr_scheduler.ExponentialLR(
            self.optimizer, gamma=config.gamma)
        # record training
        self.writer     = SummaryWriter()

    def train(self) -> None:
        self.load_checkpoint()
        while self.epoch <= self.max_epoch:
            self.train_epoch()
            self.valid_epoch()
            self.update_lr()

            if self.cpt_save_epoch:
                self.save_checkpoint("{}/{}".format(
                    self.cpt_save_path, self.epoch))
            if self.valid_loss < self.best_loss:
                self.best_loss = self.valid_loss
                self.save_checkpoint(self.cpt_save_path)

            self.epoch+=1

    def train_epoch(self) -> None:
        self.net.train()
        for i, (frames, labels) in enumerate(self.trainloader):
            # put frames and labels in GPU
            frames = frames.to(torch.float32).to(self.device)
            labels = labels.to(torch.float32).to(self.device)
            # forward
            outputs = self.net(frames)
            # loss
            loss = self.criterion(outputs, labels)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # record
            self.writer.add_scalars(
                'Loss', {'train': loss.item() / len(outputs)}, 
                (self.epoch-1)*len(self.trainloader)+i)
            self.writer.add_scalars(
                'Num', {'train': len(torch.nonzero(outputs)) / len(outputs)}, 
                (self.epoch-1)*len(self.trainloader)+i)
    
    @torch.no_grad()
    def valid_epoch(self) -> None:
        # validation
        self.net.eval()
        self.valid_loss = []
        self.valid_num = []
        for i, (frames, labels) in enumerate(self.validloader):
            # put frames and labels in GPU
            frames = frames.to(torch.float32).to(self.device)
            labels = labels.to(torch.float32).to(self.device)
            # forward
            outputs = self.net(frames)
            # loss
            loss = self.criterion(outputs, labels)
            # record
            self.valid_loss.append(loss.item() / len(outputs))
            self.valid_num.append(len(torch.nonzero(outputs)) / len(outputs))
        
        # validation loss
        self.valid_loss = torch.mean(torch.as_tensor(self.valid_loss))
        self.valid_num = torch.mean(torch.as_tensor(self.valid_num))
        
        # record
        self.writer.add_scalars(
            'Loss', {'valid': self.valid_loss}, 
            self.epoch*len(self.trainloader))
        self.writer.add_scalars(
            'Num', {'valid': self.valid_num}, 
            self.epoch*len(self.trainloader))

    @torch.no_grad()
    def update_lr(self) -> None:
        # update learning rate
        if self.scheduler.get_last_lr()[0] > 1e-7:
            self.scheduler.step()
            #self.scheduler.step(self.valid_loss)

        # record
        self.writer.add_scalar(
            'LR', self.optimizer.param_groups[0]['lr'], 
            self.epoch*len(self.trainloader))

    @torch.no_grad()
    def save_checkpoint(self, path) -> None:
        # file path checking
        if not os.path.exists(os.path.dirname(self.cpt_save_path)):
            os.makedirs(os.path.dirname(self.cpt_save_path))
        if not os.path.exists(self.cpt_save_path + "/"): 
            if self.cpt_save_epoch: os.makedirs(self.cpt_save_path)

        torch.save({
            'epoch': self.epoch,  # epoch index start from 1
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
            }, "{}.pt".format(path))

    @torch.no_grad()
    def load_checkpoint(self) -> None:
        if self.cpt_load_path == "": return
        checkpoint = torch.load("{}.pt".format(self.cpt_load_path))
        self.epoch = checkpoint['epoch']+1  # start train from next epoch index
        self.net.load_state_dict(checkpoint['net'])
        if self.cpt_load_lr:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])


if __name__ == "__main__":
    from config import Test_3 as Config
    trainer = Train(Config())
    trainer.train()
