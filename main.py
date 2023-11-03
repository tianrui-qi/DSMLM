import torch.backends.cudnn

from sml import TrainerConfig, EvaluerConfig
from sml import Trainer, Evaluer


torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True


class e14(EvaluerConfig):
    def __init__(self) -> None:
        super().__init__()
        ## ResAttUNet
        self.feats = [1, 16, 32, 64, 128]
        ## Evaluer
        self.ckpt_load_path = "ckpt/e13/200"
        self.data_save_fold = "data/e-sr/14"
        self.batch_size     = 4


class e13(TrainerConfig):
    def __init__(self) -> None:
        super().__init__()
        ## SimDataset & RawDataset
        self.scale_list = [4, 8]
        ## ResAttUNet
        self.feats = [1, 16, 32, 64, 128]
        ## Trainer
        self.ckpt_save_fold = "ckpt/e13"
        self.ckpt_load_path = "ckpt/e12/100"
        self.lr = 5e-6


class e12(TrainerConfig):
    def __init__(self) -> None:
        super().__init__()
        ## SimDataset & RawDataset
        self.scale_list = [4, 8]
        ## ResAttUNet
        self.feats = [1, 16, 32, 64, 128]
        ## Trainer
        self.max_epoch = 100
        self.ckpt_save_fold = "ckpt/e12"
        self.ckpt_load_path = "ckpt/e04/10"
        self.ckpt_load_lr   = True


if __name__ == "__main__":
    config = e14()
    if isinstance(config, TrainerConfig): Trainer(config).fit()
    if isinstance(config, EvaluerConfig): Evaluer(config).fit()
