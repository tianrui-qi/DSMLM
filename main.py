from config import Config
from dataset import SimDataset
from model import UNet2D, DeepSTORMLoss
from train import Train


if __name__ == "__main__":
    # configurations
    config = Config()
    
    # dataset
    trainset = SimDataset(config, config.num_train)
    validset = SimDataset(config, config.num_valid)

    # model and other helper for training
    net  = UNet2D()
    criterion = DeepSTORMLoss()

    trainer = Train(config, net, criterion, trainset, validset)
    trainer.train()
