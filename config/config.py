__all__ = ["Config", "ConfigTrainer", "ConfigEvaluer"]


class Config:
    def __init__(self) -> None:
        self.ckpt_disk: str = "ckpt/"
        self.data_disk: str = "data/"
        self.SimDataset = {
            "lum_info": True,  
            "dim_dst" : [160, 160, 160],

            "scale_list": [4, 8],
            "std_src": [          # std range
                [1.0, 1.0, 1.0],  # minimum std, [C, H, W], by pixel
                [3.0, 2.5, 2.5],  # maximum std, [C, H, W], by pixel
            ],
        }
        self.RawDataset  = {
            "lum_info": True,
            "dim_dst" : [160, 160, 160],

            "scale": [4, 4, 4],
            "frames_load_fold": "D:/SMLFM/hela/frames",
            "mlists_load_fold": "D:/SMLFM/hela/mlists",
        }
        self.ResAttUNet = {
            "dim"     : 3,
            "feats"   : [1, 16, 32, 64, 128],
            "use_cbam": False,
            "use_res" : False,
        }


class ConfigTrainer(Config):
    def __init__(self) -> None:
        super().__init__()
        ## Trainer
        self.max_epoch   : int = 800
        self.accumu_steps: int = 10
        # path
        self.ckpt_save_fold: str  = self.ckpt_disk + self.__class__.__name__
        self.ckpt_load_path: str  = ""        # path without .ckpt
        self.ckpt_load_lr  : bool = False     # load lr from ckpt
        # dataloader
        self.num_train : int = 10000
        self.num_valid : int = 5000
        self.batch_size: int = 1
        # optimizer
        self.lr   : float = 1e-5    # initial learning rate (lr)
        self.gamma: float = 0.95    # decay rate of lr


class ConfigEvaluer(Config):
    def __init__(self) -> None:
        super().__init__()
        ## Evaluer
        # path
        self.ckpt_load_path: str = ""   # path without .ckpts
        self.data_save_fold: str = self.data_disk + self.__class__.__name__
        # dataloader
        self.batch_size: int = 4
