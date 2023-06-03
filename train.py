import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from config import Config
from dataset import SimDataset
from model import UNet2D, DeepSTORMLoss


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience   = patience
        self.counter    = 0
        self.best_loss  = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss >= self.best_loss:
            self.counter += 1
            print(f'EarlyStopping: counter {self.counter} / {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


if __name__ == "__main__":
    # configurations
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    trainDataloader = DataLoader(
        SimDataset(config, config.num_train), batch_size=1)
    valitDataloader = DataLoader(
        SimDataset(config, config.num_valid), batch_size=1)

    net  = UNet2D()
    criterion = DeepSTORMLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)
    stopper   = EarlyStopping(config.patience)
    
    # put net and criterion in GPU
    net = net.to(device)
    criterion = criterion .to(device)

    writer = SummaryWriter()

    for epoch in range(config.epoch):
        # train
        net.train()
        for i, (frames, labels) in enumerate(trainDataloader):
            # put frames and labels in GPU
            frames = frames.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)
            # forward
            outputs = net(frames)
            # loss
            loss = criterion(outputs, labels)
            writer.add_scalars(
                'Loss', {'train': loss.item()}, epoch * len(trainDataloader) + i)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        net.eval()
        valid_epoch_loss = []
        for i, (frames, labels) in enumerate(valitDataloader):
            # put frames and labels in GPU
            frames = frames.to(torch.float32).to(device)
            labels = labels.to(torch.float32).to(device)
            # forward
            outputs = net(frames)
            # loss
            loss = criterion(outputs, labels)
            valid_epoch_loss.append(loss.item())
        loss = torch.mean(torch.as_tensor(valid_epoch_loss))
        writer.add_scalars('Loss', {'valid': loss}, len(trainDataloader))

        # early stopping
        stopper(loss)
        if stopper.early_stop: break

        # update learning rate
        scheduler.step()

        # save checkpoint
        torch.save({
            'epoch':epoch,
            'net':net.state_dict(),
            'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict()
            }, config.checkpoint_path)
