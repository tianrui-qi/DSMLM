import torch.backends.cudnn

import argparse

import sml

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


""" 
Currently we gauss the problem in e01-03 is because the complexcity of the 
network is not enough to locolize. Reference to d05 and other, it's seems like
these checkbox is cause by not enough training. Thus, we try to increase the
complexcity of the network and retrain.
"""


class e04(sml.ConfigTrainer):
    def __init__(self) -> None:
        super().__init__()
        ## SimDataset & RawDataset
        self.lum_info   = False
        self.scale_list = [4]
        ## ResAttUNet
        self.feats = [1, 16, 32, 64, 128]
        ## Train
        self.ckpt_save_fold = "ckpt/e04"
        self.lr = 5e-5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, choices=["train", "eval"])
    args = parser.parse_args()

    config = e04()
    if   args.mode == "train": sml.Trainer(config).train()
    elif args.mode == "eval" : sml.Evaluator(config).eval()
    else: parser.print_help()
