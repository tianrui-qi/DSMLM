import torch.backends.cudnn

import argparse

from sml import ConfigTrain, ConfigEval, Trainer, Evaluator


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class e_02(ConfigTrain):
    def __init__(self) -> None:
        super().__init__()
        ## Train
        #self.lr = 1e-5
        self.lr = 1e-4
        #self.ckpt_load_path = "ckpt/e_01/150"
        self.ckpt_load_path = "ckpt/e_02/160"
        self.ckpt_save_fold = "ckpt/e_02"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, choices=["train", "eval"])
    args = parser.parse_args()

    config = e_02()
    if   args.mode == "train": Trainer(config).train()
    elif args.mode == "eval" : Evaluator(config).eval()
    else: parser.print_help()
