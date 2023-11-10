import torch.backends.cudnn

from sml import TrainerConfig, EvaluerConfig
from sml import Trainer, Evaluer


torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True


class e19(EvaluerConfig):
    def __init__(self) -> None:
        super().__init__()
        self.scale = [4, 8, 8]
        ## ResAttUNet
        self.feats = [1, 16, 32, 64, 128, 256]
        ## Evaluer
        self.ckpt_load_path = "ckpt/e17/400"
        self.data_save_fold = "data/e19"


class e18(EvaluerConfig):
    def __init__(self) -> None:
        super().__init__()
        ## ResAttUNet
        self.feats = [1, 16, 32, 64, 128, 256]
        ## Evaluer
        self.ckpt_load_path = "ckpt/e17/400"
        self.data_save_fold = "data/e18"


if __name__ == "__main__":
    config = e18()
    if isinstance(config, TrainerConfig): Trainer(config).fit()
    if isinstance(config, EvaluerConfig): Evaluer(config).fit()
