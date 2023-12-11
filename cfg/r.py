import argparse

from .config import *

__all__ = ["r"]


class r(Config):
    def __init__(self) -> None:
        super().__init__("evalu")
        # get necessary arguments
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-s", type=int, required=True, dest="scale",
            help="Scale up factor, 4 or 8. " + 
            "When scale up by 4 or 8, the code will automatically load the " +
            "corresponding checkpoint from `ckpt/e08/340.ckpt` or " + 
            "`ckpt/e10/450.ckpt`."
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
            "-b", type=int, required=True, dest="batch_size",
            help="Batch size. Set this value according to your GPU memory."
        )
        args = parser.parse_args()
        # set configurations
        self.RawDataset["num"] = None
        if args.scale not in [4, 8]: raise ValueError("scale must be 4 or 8")
        if args.scale == 4: 
            self.RawDataset["scale"] = [4, 4, 4]
            self.Evaluer["ckpt_load_path"] = "ckpt/e08/340"
        if args.scale == 8: 
            self.RawDataset["scale"] = [4, 8, 8]
            self.Evaluer["ckpt_load_path"] = "ckpt/e10/450"
        self.RawDataset["frames_load_fold"] = args.frames_load_fold
        self.Evaluer["data_save_fold"] = args.data_save_fold
        self.Evaluer["batch_size" ] = args.batch_size
        self.Evaluer["num_workers"] = args.batch_size*4
