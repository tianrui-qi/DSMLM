import torch.backends.cudnn

from sml import TrainerConfig, EvaluerConfig
from sml import Trainer, Evaluer


torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True


class e16(TrainerConfig):
    def __init__(self) -> None:
        super().__init__()
        ## SimDataset & RawDataset
        self.lum_info = False
        self.scale_list = [4]
        ## ResAttUNet
        self.feats = [1, 32, 64, 128, 256, 512]
        ## Trainer
        self.ckpt_save_fold = "ckpt/e16"


if __name__ == "__main__":
    config = e16()
    if isinstance(config, TrainerConfig): Trainer(config).fit()
    if isinstance(config, EvaluerConfig): Evaluer(config).fit()
