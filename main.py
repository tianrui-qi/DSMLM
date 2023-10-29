import torch.backends.cudnn

import sml

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True


class e12(sml.EvaluerConfig):
    def __init__(self) -> None:
        super().__init__()
        self.scale = [4, 8, 8]
        ## Evaluer
        self.ckpt_load_path = "ckpt/e10/320"
        self.data_save_fold = "data/e-sr/12"
        self.batch_size     = 4


if __name__ == "__main__":
    config = e12()
    if isinstance(config, sml.TrainerConfig): sml.Trainer(config).fit()
    if isinstance(config, sml.EvaluerConfig): sml.Evaluer(config).fit()
