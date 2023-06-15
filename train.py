import os  # for file checking
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from config import Config
from dataset import SimDataset
from model import UNet2D
from criterion import Criterion


class Train:
    def __init__(self, config, net, criterion, trainset, validset):
        # configurations
        # for train
        self.max_epoch  = config.max_epoch
        self.batch_size = config.batch_size
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # for runnning log
        self.logdir     = config.logdir
        # for checkpoint
        self.load       = config.load
        self.checkpoint_path = config.checkpoint_path
        self.save_pt_epoch   = config.save_pt_epoch
        
        # index
        self.epoch      = 1    # epoch index start from 1
        self.stage      = "1"  # idx for stage, 1, 12, or 2
        self.stage_idx  = 0    # idx for number of epoch in stage 12
        self.valid_loss = torch.tensor(float("inf"))
        self.best_loss  = torch.tensor(float("inf"))
        self.valid_num  = 0

        # dataloader
        self.trainloader = self.dataloader(trainset)
        self.validloader = self.dataloader(validset)
        # model
        self.net        = net.to(self.device)
        self.criterion  = criterion.to(self.device)
        
        # optimizer
        self.optimizer  = optim.Adam(self.net.parameters(), lr=config.lr)
        self.scheduler  = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=config.patience)
        # record training
        self.writer     = SummaryWriter(log_dir=self.logdir)        

    def dataloader(self, dataset):
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.batch_size, 
            pin_memory=True
            )

    def train(self):
        # epoch index start from 1
        if self.load: self.load_checkpoint()
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
                'Loss', {'train': loss.item()}, 
                (self.epoch-1)*len(self.trainloader)+i)
            self.writer.add_scalars(
                'Num', {'train': len(torch.nonzero(outputs))}, 
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
            self.valid_loss.append(loss.item())
            self.valid_num.append(float(len(torch.nonzero(outputs))))
        
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
        # decide the stage and learning rate
        if self.stage == "1" and self.valid_num < 1000:
            self.stage = "12"
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.00001
        if self.stage == "12": self.stage_idx += 1
        if self.stage == "12" and self.stage_idx > 5: 
            self.stage = "2"
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0001
        if self.stage == "2": self.scheduler.step(self.valid_loss)

        # record
        self.writer.add_scalars(
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
    def load_checkpoint(self):
        checkpoint = torch.load("{}.pt".format(self.checkpoint_path))
        self.epoch = checkpoint['epoch']+1  # start train from next epoch index
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])


if __name__ == "__main__":
    # configurations
    config = Config()
    config.logdir = "runs/train"  # type: ignore
    config.checkpoint_path = "checkpoints/train"
    config.save_pt_epoch = True
    # dataset
    trainset = SimDataset(config, config.num_train)
    validset = SimDataset(config, config.num_valid)
    # model and other helper for training
    net       = UNet2D(config)
    criterion = Criterion(config)
    # train
    trainer = Train(config, net, criterion, trainset, validset)
    trainer.train()
