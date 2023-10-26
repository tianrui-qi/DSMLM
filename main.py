import torch.backends.cudnn

import sml

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True


class e08(sml.EvaluerConfig):
    def __init__(self) -> None:
        super().__init__()
        ## SimDataset & RawDataset
        self.lum_info = False
        ## ResAttUNet
        self.feats = [1, 16, 32, 64, 128]
        ## Evaluer
        self.ckpt_load_path = "ckpt/e06/90"
        self.data_save_fold = "data/e-sr/08"


if __name__ == "__main__":
    config = e08()
    if isinstance(config, sml.TrainerConfig): sml.Trainer(config).fit()
    if isinstance(config, sml.EvaluerConfig): sml.Evaluer(config).fit()
