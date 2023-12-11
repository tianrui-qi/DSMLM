import torch
import torch.cuda
import torch.backends.cudnn

import numpy as np
import random
import os

import cfg, sml


torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    np.random.seed(seed)
    random.seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)  # python hash seed


if __name__ == "__main__":
    set_seed(42)
    config = cfg.r()
    if config.mode == "train": sml.Trainer(
        **config.Trainer,
        trainset=sml.SimDataset(**config.SimDataset), 
        validset=sml.RawDataset(**config.RawDataset), 
        model=sml.ResAttUNet(**config.ResAttUNet), 
    ).fit()
    if config.mode == "evalu": sml.Evaluer(
        **config.Evaluer,
        evaluset=sml.RawDataset(**config.RawDataset), 
    ).fit()
