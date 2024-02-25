__all__ = ["ConfigEvaluer", "ConfigTrainer"]


class ConfigEvaluer:
    def __init__(self, args) -> None:
        self.evaluset = {
            "num": None,    # must be None
            "lum_info": True,
            "dim_dst" : [160, 160, 160],

            "scale": [4, args.scale, args.scale],
            "rng_sub_user": args.rng_sub_user,
            "frames_load_fold": args.frames_load_fold,
            "mlists_load_fold": None,   # not used
        }
        self.runner = {
            # path
            "data_save_fold": args.data_save_fold,
            "ckpt_load_path": args.ckpt_load_path,  # path without .ckpt
            "temp_save_fold": args.temp_save_fold,
            # drift
            "stride": args.stride,  # unit: frames
            "window": args.window,  # unit: frames
            "method": args.method,  # DCC, MCC, or RCC
            # data
            "batch_size": args.batch_size,
        }


class ConfigTrainer:
    def __init__(self) -> None:
        self.trainset = {
            "num": 10000,
            "lum_info": True,  
            "dim_dst" : [160, 160, 160],

            "scale_list": [4, 8],
            "std_src": [    # std range, [min, max] std, [C, H, W], by pixel
                [1.0, 1.0, 1.0], [3.0, 2.5, 2.5]
            ],
        }
        self.validset = {
            "num": 5000,
            "lum_info": True,
            "dim_dst" : [160, 160, 160],

            "scale": [4, 4, 4],
            "rng_sub_user": None,   # not used for training
            "frames_load_fold": "D:/frames",
            "mlists_load_fold": "D:/mlists",
        }
        self.model = {
            "dim"  : 3,
            "feats": [1, 16, 32, 64, 128],
        }
        self.runner = {
            # train
            "max_epoch": 800,
            "accumu_steps": 10,
            # checkpoint
            "ckpt_save_fold": "ckpt/" + self.__class__.__name__,
            "ckpt_load_path": "",       # path without .ckpt
            "ckpt_load_lr"  : False,    # load lr from ckpt
            # data
            "batch_size": 1,
            # optimizer
            "lr": 1e-5,     # initial learning rate (lr)
        }
