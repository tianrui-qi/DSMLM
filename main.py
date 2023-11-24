import torch.backends.cudnn

import numpy as np
import random
import os

import sml
import config


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
    for cfg in (
        config.e08(),
    ):
        if isinstance(cfg, config.TrainerConfig): sml.Trainer(cfg).fit()
        if isinstance(cfg, config.EvaluerConfig): sml.Evaluer(cfg).fit()
