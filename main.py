import torch
import torch.cuda
import torch.backends.cudnn

import numpy as np

import random
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='cupy')

import cfg, src

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
    if isinstance(config, cfg.ConfigEvaluer): src.Evaluer(
        **config.runner,
        evaluset=src.RawDataset(**config.evaluset), 
    ).fit()
    if isinstance(config, cfg.ConfigTrainer): src.Trainer(
        **config.runner,
        trainset=src.SimDataset(**config.trainset), 
        validset=src.RawDataset(**config.validset), 
        model=src.ResAttUNet(**config.model), 
    ).fit()
