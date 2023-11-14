import torch.backends.cudnn

from sml import TrainerConfig, EvaluerConfig
from sml import Trainer, Evaluer


torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True


"""
features number : [1, 32, 64, 128, 256, 512, 1024]
Trainable paras : 85,591,649
Training   speed: 0.58 steps /s ( 10 iterations/step)
Validation speed: 0.91 steps /s ( 10 iterations/step)
Evaluation speed:      frames/s ( 16 subframes/frame)
                       frames/s ( 64 subframes/frame)
"""


class e19(TrainerConfig):
    def __init__(self) -> None:
        super().__init__()
        ## data
        self.scale_list = [4, 8, 16]
        ## model
        self.feats = [1, 32, 64, 128, 256, 512, 1024]
        ## runner
        self.ckpt_save_fold = "ckpt/e19"
        self.ckpt_load_path = "ckpt/e18/230"


class e18(TrainerConfig):
    def __init__(self) -> None:
        super().__init__()
        ## data
        self.lum_info = False
        self.scale_list = [4, 8, 16]
        ## model
        self.feats = [1, 32, 64, 128, 256, 512, 1024]
        ## runner
        self.ckpt_save_fold = "ckpt/e18"
        self.ckpt_load_path = "ckpt/e17/170"


class e17(TrainerConfig):
    def __init__(self) -> None:
        super().__init__()
        ## data
        self.lum_info = False
        self.scale_list = [4]
        ## model
        self.feats = [1, 32, 64, 128, 256, 512, 1024]
        ## runner
        self.ckpt_save_fold = "ckpt/e17"
        self.ckpt_load_path = "ckpt/e16/160"


class e16(TrainerConfig):
    def __init__(self) -> None:
        super().__init__()
        ## data
        self.lum_info = False
        self.scale_list = [4]
        ## model
        self.feats = [1, 32, 64, 128, 256]
        ## runner
        self.ckpt_save_fold = "ckpt/e16"
        self.ckpt_load_path = "ckpt/d02/140"


if __name__ == "__main__":
    config = e19()
    if isinstance(config, TrainerConfig): Trainer(config).fit()
    if isinstance(config, EvaluerConfig): Evaluer(config).fit()
