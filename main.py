import torch.backends.cudnn

import argparse

from sml import ConfigTrain, ConfigEval, Trainer, Evaluator


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


""" 
Currently we gauss the problem in e01-03 is because the complexcity of the 
network is not enough to locolize. Reference to d05 and other, it's seems like
these checkbox is cause by not enough training. Thus, we try to increase the
complexcity of the network and retrain.
"""


class e05(ConfigTrain):
    def __init__(self) -> None:
        super().__init__()
        self.feats = [1, 16, 32, 64, 128]

        self.scale_list = [4]

        self.lr = 1e-5
        self.ckpt_save_fold = "ckpt/e05"
        self.ckpt_load_path = "ckpt/e04/10"


class e04(ConfigTrain):
    def __init__(self) -> None:
        super().__init__()
        self.feats = [1, 16, 32, 64, 128]

        self.ckpt_save_fold = "ckpt/e04"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, choices=["train", "eval"])
    args = parser.parse_args()

    config = e05()
    if   args.mode == "train": Trainer(config).train()
    elif args.mode == "eval" : Evaluator(config).eval()
    else: parser.print_help()
