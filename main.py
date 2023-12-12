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
    config = cfg.ConfigEvaluer()
    if isinstance(config, cfg.ConfigEvaluer): sml.Evaluer(
        **config.runner,
        evaluset=sml.RawDataset(**config.evaluset), 
    ).fit()
    if isinstance(config, cfg.ConfigTrainer): sml.Trainer(
        **config.runner,
        trainset=sml.SimDataset(**config.trainset), 
        validset=sml.RawDataset(**config.validset), 
        model=sml.ResAttUNet(**config.model), 
    ).fit()
