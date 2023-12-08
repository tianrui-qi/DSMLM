import torch.backends.cudnn

import numpy as np
import random
import os

import argparse

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


class temp(config.ConfigEvaluer):
    def __init__(self) -> None:
        super().__init__()
        # get three necessary arguments we need
        parser = argparse.ArgumentParser(
            description="Run script with command line arguments."
        )
        parser.add_argument(
            "-L", type=str, required=True, dest="frames_load_fold",
            help="Path to the frames load folder."
        )
        parser.add_argument(
            "-S", type=str, required=True, dest="data_save_fold",
            help="Path to the data save folder."
        )
        parser.add_argument(
            "-s", type=int, required=True, dest="scale",
            help="Scale up factor, 4 or 8. " + 
            "When scale up by 4 or 8, the code will automatically load the " +
            "corresponding checkpoint from `ckpt/e04/340.ckpt` or " + 
            "`ckpt/e08/340.ckpt`."
        )
        parser.add_argument(
            "-b", type=int, required=True, dest="batch_size",
            help="Batch size. Set this value according to your GPU memory."
        )
        args = parser.parse_args()
        # set configuration
        self.RawDataset["frames_load_fold"] = args.frames_load_fold
        if args.scale not in [4, 8]:
            raise ValueError("scale must be 4 or 8")
        if args.scale == 4:
            self.ckpt_load_path = self.ckpt_disk + "e08/340"
        if args.scale == 8:
            self.RawDataset["scale"] = [4, 8, 8]
            self.ResAttUNet["feats"] = [1, 16, 32, 64, 128, 256, 512]
            self.ckpt_load_path = self.ckpt_disk + "e10/340"
        self.data_save_fold = args.data_save_fold
        self.batch_size = args.batch_size


if __name__ == "__main__":
    set_seed(42)
    cfg = config.e10()
    if isinstance(cfg, config.ConfigTrainer): sml.Trainer(cfg).fit()
    if isinstance(cfg, config.ConfigEvaluer): sml.Evaluer(cfg).fit()
