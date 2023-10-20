import torch.backends.cudnn

import argparse

from sml import Config, Trainer, Evaluator


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


class e_01(Config):
    def train(self) -> None:
        super().train()
        ## Train
        self.lr = 1e-5
        self.ckpt_load_path = "ckpt/d_04/140"
        self.ckpt_save_fold = "ckpt/e_01"

    def eval(self) -> None: return NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, choices=["train", "eval"])
    args = parser.parse_args()

    config = e_01()
    if args.mode == "train":
        config.train()
        Trainer(config).train()
    elif args.mode == "eval":
        config.eval()
        Evaluator(config).eval()
    else:
        parser.print_help()
