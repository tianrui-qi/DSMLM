import torch
import torch.cuda
import torch.backends.cudnn

import numpy as np
import random
import os

import config, sml


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
    cfg = config.temp()
    if cfg.mode == "train": 
        sml.Trainer(
            **cfg.Trainer,
            trainset=sml.SimDataset(**cfg.SimDataset), 
            validset=sml.RawDataset(**cfg.RawDataset), 
            model=sml.ResAttUNet(**cfg.ResAttUNet), 
        ).fit()
    if cfg.mode == "evalu":
        cfg.RawDataset["num"] = None
        sml.Evaluer(
            **cfg.Evaluer,
            evaluset=sml.RawDataset(**cfg.RawDataset), 
            model = sml.ResAttUNet(**cfg.ResAttUNet)
        ).fit()
