import torch.backends.cudnn

from sml import TrainerConfig, EvaluerConfig
from sml import Trainer, Evaluer


torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True


class e17(TrainerConfig):
    def __init__(self) -> None:
        super().__init__()
        ## ResAttUNet
        self.feats = [1, 16, 32, 64, 128, 256]
        ## Trainer
        self.ckpt_save_fold = "ckpt/e17"
        self.ckpt_load_path = "ckpt/e16/300"
        self.lr = 1e-6


class e16(TrainerConfig):
    def __init__(self) -> None:
        super().__init__()
        ## ResAttUNet
        self.feats = [1, 16, 32, 64, 128, 256]
        ## Trainer
        self.ckpt_save_fold = "ckpt/e16"
        self.ckpt_load_path = "ckpt/e13/200"
        self.lr = 1e-5


if __name__ == "__main__":
    config = e17()
    if isinstance(config, TrainerConfig): Trainer(config).fit()
    if isinstance(config, EvaluerConfig): Evaluer(config).fit()
