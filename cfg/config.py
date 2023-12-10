__all__ = ["Config"]


class Config:
    def __init__(self, mode: str) -> None:
        self.mode: str = mode    # "train" or "evalu"
        self.ckpt_disk: str = "ckpt/"
        self.data_disk: str = "data/"
        self.SimDataset = {
            "num": 10000,
            "lum_info": True,  
            "dim_dst" : [160, 160, 160],

            "scale_list": [4, 8],
            "std_src": [          # std range
                [1.0, 1.0, 1.0],  # minimum std, [C, H, W], by pixel
                [3.0, 2.5, 2.5],  # maximum std, [C, H, W], by pixel
            ],
        }
        self.RawDataset  = {
            "num": 5000,
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
        self.Trainer = {
            "max_epoch": 800,
            "accumu_steps": 10,
            # path
            "ckpt_save_fold": self.ckpt_disk + self.__class__.__name__,
            "ckpt_load_path": "",       # path without .ckpt
            "ckpt_load_lr"  : False,    # load lr from ckpt
            # data
            "batch_size": 1,
            "num_workers": 4,
            # optimizer
            "lr"   : 1e-5,  # initial learning rate (lr)
            "gamma": 0.95,  # decay rate of lr
        }
        self.Evaluer = {
            # path
            "ckpt_load_path": "",       # path without .ckpt
            "data_save_fold": self.data_disk + self.__class__.__name__,
            # data
            "batch_size": 4,
            "num_workers": 16,
        }
