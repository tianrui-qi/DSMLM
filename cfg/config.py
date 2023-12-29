import argparse

__all__ = ["ConfigEvaluer", "ConfigTrainer"]


class ConfigEvaluer:
    def __init__(self) -> None:
        args = self._getArgument()
        self.evaluset = {
            "num": None,    # must be None
            "lum_info": True,
            "dim_dst" : [160, 160, 160],

            "scale": [4, args.scale, args.scale],
            "frames_load_fold": args.frames_load_fold,
            "mlists_load_fold": None,   # not used
        }
        self.runner = {
            # path
            "data_save_fold": args.data_save_fold,
            "ckpt_load_path": args.ckpt_load_path,  # path without .ckpt
            # drift
            "stride": args.stride,  # unit: frames
            "window": args.window,  # unit: frames
            # data
            "batch_size": args.batch_size,
        }

    def _getArgument(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-s", type=int, required=False, dest="scale", 
            choices=[4, 8], default=4,
            help="Scale up factor, 4 or 8. Default: 4."
        )
        parser.add_argument(
            "-L", type=str, required=True, dest="frames_load_fold",
            help="Path to the frames load folder."
        )
        parser.add_argument(
            "-S", type=str, required=True, dest="data_save_fold",
            help="Path to the data save folder."
        )
        parser.add_argument(
            "-C", type=str, required=False, dest="ckpt_load_path",
            help="Path to the checkpoint load file without .ckpt. " +
            "Default: `ckpt/e08/340` or `ckpt/e10/450` " + 
            "when scale up factor is 4 or 8."
        )
        parser.add_argument(
            "-stride", type=int, required=False, dest="stride", 
            default=0,
            help="Step size of the drift corrector, unit frames. Default: 0."
        )
        parser.add_argument(
            "-window", type=int, required=False, dest="window", 
            default=0,
            help="Number of frames in each window, unit frames. Default: 0."
        )
        parser.add_argument(
            "-b", type=int, required=True, dest="batch_size",
            help="Batch size. Set this value according to your GPU memory."
        )
        args = parser.parse_args()
        # set default value for ckpt_load_path
        if args.ckpt_load_path is None:
            if args.scale == 4: args.ckpt_load_path = "ckpt/e08/340"
            if args.scale == 8: args.ckpt_load_path = "ckpt/e10/450"
        return args


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
            "frames_load_fold": "D:/SMLFM/hela/frames",
            "mlists_load_fold": "D:/SMLFM/hela/mlists",
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
