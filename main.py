import torch.backends.cudnn

import sml

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True


class e11(sml.EvaluerConfig):
    def __init__(self) -> None:
        super().__init__()
        ## Evaluer
        self.ckpt_load_path = "ckpt/e10/320"
        self.data_save_fold = "data/e-sr/11"
        self.batch_size     = 4


class e10(sml.TrainerConfig):
    def __init__(self) -> None:
        super().__init__()
        ## SimDataset & RawDataset
        self.scale_list = [4, 8]
        ## Trainer
        self.ckpt_save_fold = "ckpt/e10"
        self.ckpt_load_path = "ckpt/e09/240"


class e09(sml.TrainerConfig):
    def __init__(self) -> None:
        super().__init__()
        ## SimDataset & RawDataset
        self.lum_info = False
        self.scale_list = [4, 8]
        ## Trainer
        self.ckpt_save_fold = "ckpt/e09"
        self.ckpt_load_path = "ckpt/d04/140"


if __name__ == "__main__":
    config = e11()
    if isinstance(config, sml.TrainerConfig): sml.Trainer(config).fit()
    if isinstance(config, sml.EvaluerConfig): sml.Evaluer(config).fit()
