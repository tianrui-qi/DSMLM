import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter


class Train:
    def __init__(
            self, config, net, criterion, trainset, validset
            ) -> None:
        # configurations
        self.num_train  = config.num_train
        self.num_valid  = config.num_valid
        self.max_epoch  = config.max_epoch
        self.batch_size = config.batch_size
        self.lr         = config.lr
        self.gamma      = config.gamma
        self.patience   = config.patience
        self.checkpoint_path = config.checkpoint_path
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        
        # index
        self.epoch      = 0
        self.counter    = 0                             # for early stopping
        self.valid_loss = torch.tensor(float("inf"))    # for early stopping
        self.best_loss  = torch.tensor(float("inf"))    # for early stopping
        self.stop       = False                         # for early stopping

        # dataloader
        self.trainloader = self.dataloader(trainset)
        self.validloader = self.dataloader(validset)
        # model
        self.net        = net.to(self.device)
        self.criterion  = criterion.to(self.device)
        
        # optimizer
        self.optimizer  = optim.Adam(self.net.parameters(), lr=config.lr)
        self.scheduler  = lr_scheduler.ExponentialLR(
            self.optimizer, gamma=config.gamma)
        # record training
        self.writer     = SummaryWriter()        

    def dataloader(self, dataset):
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.batch_size, 
            pin_memory=True
            )

    def train(self, load=True):
        if load: self.load_checkpoint()
        while self.stop is False and self.epoch < self.max_epoch:
            self.train_epoch()
            self.valid_epoch()
            self.early_stop()
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
                self.epoch*len(self.trainloader)+i)
        # update learning rate
        self.scheduler.step()  
    
    @torch.no_grad()
    def valid_epoch(self):
        # validation
        self.net.eval()
        self.valid_loss = []
        for i, (frames, labels) in enumerate(self.validloader):
            # put frames and labels in GPU
            frames = frames.to(torch.float32).to(self.device)
            labels = labels.to(torch.float32).to(self.device)
            # forward
            outputs = self.net(frames)
            # loss
            loss = self.criterion(outputs, labels)
            self.valid_loss.append(loss.item())
        self.valid_loss = torch.mean(torch.as_tensor(self.valid_loss))
        self.writer.add_scalars(
            'Loss', {'valid': self.valid_loss}, 
            (self.epoch+1)*len(self.trainloader))
    
    @torch.no_grad()
    def early_stop(self):
        if self.valid_loss >= self.best_loss:
            self.counter += 1
            if self.counter >= self.patience: self.stop = True
        else:
            self.best_loss = self.valid_loss
            self.counter = 0
            self.save_checkpoint()

    @torch.no_grad()
    def save_checkpoint(self):
        torch.save({
            'epoch': self.epoch,
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
            }, self.checkpoint_path)

    @torch.no_grad()
    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.epoch = checkpoint['epoch'] + 1
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
