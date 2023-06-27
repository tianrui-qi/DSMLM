import os  # for file checking
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard.writer import SummaryWriter

from data import getData
from model import *


class Train:
    def __init__(self, config, dataloader):
        # configurations
        # for train
        self.max_epoch  = config.max_epoch
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # for checkpoint
        self.checkpoint_path = config.checkpoint_path
        self.save_pt_epoch   = config.save_pt_epoch
        
        # index
        self.epoch      = 1    # epoch index start from 1
        self.valid_loss = torch.tensor(float("inf"))
        self.best_loss  = torch.tensor(float("inf"))
        self.valid_num  = 0

        # data
        self.trainloader = dataloader["train"]
        self.validloader = dataloader["valid"]
        # model
        self.net        = UNet2D(config).to(self.device)
        self.criterion  = Criterion(config).to(self.device)
        
        # optimizer
        self.optimizer  = optim.Adam(self.net.parameters(), lr=config.lr)
        self.scheduler  = lr_scheduler.ExponentialLR(
            self.optimizer, gamma=config.gamma)
        # record training
        self.writer     = SummaryWriter()        

    def train(self):
        while self.epoch <= self.max_epoch:
            self.train_epoch()
            self.valid_epoch()
            self.update_lr()

            if self.save_pt_epoch:
                self.save_checkpoint("{}/{}".format(
                    self.checkpoint_path, self.epoch))
            if self.valid_loss < self.best_loss:
                self.best_loss = self.valid_loss
                self.save_checkpoint(self.checkpoint_path)

            self.epoch+=1

    def train_epoch(self):
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
    def valid_epoch(self):
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
    def update_lr(self):
        # update learning rate
        if self.scheduler.get_last_lr()[0] > 1e-7:
            self.scheduler.step()
            #self.scheduler.step(self.valid_loss)

        # record
        self.writer.add_scalar(
            'LR', self.optimizer.param_groups[0]['lr'], 
            self.epoch*len(self.trainloader))

    @torch.no_grad()
    def save_checkpoint(self, path):
        # file path checking
        if not os.path.exists(os.path.dirname(self.checkpoint_path)):
            os.makedirs(os.path.dirname(self.checkpoint_path))
        if not os.path.exists(self.checkpoint_path + "/"): 
            if self.save_pt_epoch: os.makedirs(self.checkpoint_path)

        torch.save({
            'epoch': self.epoch,  # epoch index start from 1
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
            }, "{}.pt".format(path))

    @torch.no_grad()
    def load_checkpoint(self, path, load_lr = True):
        checkpoint = torch.load("{}.pt".format(path))
        self.epoch = checkpoint['epoch']+1  # start train from next epoch index
        self.net.load_state_dict(checkpoint['net'])
        if load_lr:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])


if __name__ == "__main__":
    from config import Config
    config = Config()
    # learning rate
    config.lr = 0.00001
    # checkpoint
    config.checkpoint_path = "checkpoints/test_8"
    config.save_pt_epoch = True
    # data
    config.num = [3000, 900]
    config.batch_size  = 3
    config.num_workers = 3
    dataloader = getData(config, "simu", True)
    # train
    trainer = Train(config, dataloader)
    #trainer.load_checkpoint("checkpoints/test_9", load_lr=False)
    trainer.train()
