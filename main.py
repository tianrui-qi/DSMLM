import torch.backends.cudnn

import argparse

from sml import ConfigTrain, ConfigEval, Trainer, Evaluator


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class e02(ConfigTrain):
    def __init__(self) -> None:
        super().__init__()
        ## Train
        #self.lr = 1e-5
        #self.ckpt_load_path = "ckpt/e01/150"
        self.ckpt_load_path = "ckpt/e02/167"
        self.ckpt_save_fold = "ckpt/e02"
        self.ckpt_load_lr   = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, choices=["train", "eval"])
    args = parser.parse_args()

    config = e02()
    if   args.mode == "train": Trainer(config).train()
    elif args.mode == "eval" : Evaluator(config).eval()
    else: parser.print_help()
