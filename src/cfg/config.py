__all__ = ["ConfigEvaluer", "ConfigTrainer"]


class ConfigEvaluer:
    def __init__(
        self, scale: int | None, rng_sub_user: list[int] | None, 
        frames_load_fold: str,
        data_save_fold: str | None, ckpt_load_path: str | None, 
        temp_save_fold: str | None, 
        stride: int | None, window: int | None, method: str | None, 
        batch_size: int, num_workers: int, **kwargs,
    ) -> None:
        self.evaluset = {   # src.data.RawDataset
            "num": None,    # must be None
            "lum_info": True,
            "dim_dst" : [160, 160, 160],

            "scale": [4, scale, scale],
            "rng_sub_user": rng_sub_user,
            "frames_load_fold": frames_load_fold,
            "mlists_load_fold": None,   # not used
        }
        self.runner = {     # src.runner.Evaluer
            # path
            "data_save_fold": data_save_fold,
            "ckpt_load_path": ckpt_load_path,
            "temp_save_fold": temp_save_fold,
            # drift
            "stride": stride,   # unit: frames
            "window": window,   # unit: frames
            "method": method,   # DCC, MCC, or RCC
            # data
            "batch_size": batch_size,
            "num_workers": num_workers,
        }


class ConfigTrainer:
    def __init__(self) -> None:
        self.trainset = {   # src.data.SimDataset
            "num": 10000,
            "lum_info": True,  
            "dim_dst" : [160, 160, 160],

            "scale_list": [4, 8],
            "std_src": [    # std range, [min, max] std, [C, H, W], by pixel
                [1.0, 1.0, 1.0], [3.0, 2.5, 2.5]
            ],
            "psf_load_path": "",    # empty to disable convolve, data/psf.tif
        }
        self.validset = {   # src.data.RawDataset
            "num": 5000,
            "lum_info": True,
            "dim_dst" : [160, 160, 160],

            "scale": [4, 4, 4],
            "rng_sub_user": None,   # not used for training
            "frames_load_fold": "C:/Users/tianrui/Desktop/tubulin/frames",
            "mlists_load_fold": "C:/Users/tianrui/Desktop/tubulin/mlists",
        }
        self.model = {      # src.model.ResAttUNet
            "dim"  : 3,
            "feats": [1, 16, 32, 64, 128],
        }
        self.runner = {     # src.runner.Trainer
            # train
            "max_epoch": 800,
            "accumu_steps": 10,
            # checkpoint
            "ckpt_save_fold": "ckpt/" + self.__class__.__name__,
            "ckpt_load_path": "",
            "ckpt_load_lr"  : False,    # load lr from ckpt
            # data
            "batch_size" : 1,
            "num_workers": 4,
            # optimizer
            "lr": 1e-5,     # initial learning rate (lr)
        }
