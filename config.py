from typing import List


class Config:
    def __init__(self) -> None:
        # dimensional config
        # MUST be same accross whole pipline
        self.dim_frame: List[int] = [64, 64, 64]    # [C, H, W], by pixel
        self.up_sample: List[int] = [ 2,  4,  4]    # [C, H, W], by scale

        # =============================== train ============================== #

        ## (Class) Train
        # train
        self.device: str = "cuda"
        self.max_epoch: int = 500
        self.accumu_steps: int = 20  # unit: batch
        # learning rate
        self.lr   : float = 1e-3    # initial learning rate (lr)
        self.gamma: float = 0.95
        # checkpoint
        self.ckpt_save_folder: str  = "ckpt"    # folder store ckpt every epoch
        self.ckpt_load_path  : str  = ""        # path without .ckpt
        self.ckpt_load_lr    : bool = False     # load lr from ckpt

        # =============================== eval =============================== #

        ## (class) Eval - also use some config of train and data
        # train: self.device, self.ckpt_load_path
        # eval
        self.outputs_save_path: str = "data/outputs"   # path without .tif
        self.labels_save_path : str = "data/labels"    # path without .tif
        # data: self.num_sub, self.batch_size

        # =============================== model ============================== #

        ## (class) Criterion
        self.kernel_size : int   = 7    # kernel size of GaussianBlur
        self.kernel_sigma: float = 1.0  # sigma of kernel

        # =============================== data =============================== #

        ## (class) SimDataset
        # config for adjust distribution of molecular
        self.mol_range: List[int] = [0, 64]     # min, max num of mol/frame
        self.std_range: List[List[float]] = [   # std range of each dimension
            [2.6, 1.0, 1.0],  # FWHM [400 300 300], pixel size [65 130 130]
            [5.9, 2.0, 2.0],  # FWHM [900 600 600], pixel size [65 130 130]
        ]
        self.lum_range: List[float] = [0.0, 1.0]
        # config for reducing resolution and adding noise
        self.bitdepth   : int   = 16
        self.qe         : float = 0.82
        self.sensitivity: float = 5.88
        self.dark_noise : float = 2.29

        ## (class) RawDataset
        # subframe index
        self.h_range: List[int] = [0, 9]
        self.w_range: List[int] = [0, 9]
        self.num_sub: int = 100
        # data path
        self.frames_folder = "D:/frames"
        self.mlists_folder = "D:/mlists"

        ## (def) getDataLoader
        self.num : List[int] = [5000 , 5000 ]
        self.type: List[str] = ["Sim", "Raw"]
        self.batch_size : int = 1
        self.num_workers: int = 1


class ConfigTrain(Config):
    def __init__(self) -> None:
        super().__init__()


class ConfigEval(ConfigTrain):
    def __init__(self) -> None:
        super().__init__()
        ## (class) Eval
        checkpoint = 1
        self.ckpt_load_path = "{}/{}".format(self.ckpt_save_folder, checkpoint)
        self.outputs_save_path = "data/outputs_{}".format(checkpoint)
        self.labels_save_path  = "data/labels_{}".format(checkpoint) 
        ## (class) RawDataset
        self.h_range = [5, 8]
        self.w_range = [6, 9]
        self.num_sub = 16
        ## (def) getDataLoader
        self.num  = [1000 * self.num_sub]
        self.type = ["Raw"]
        self.batch_size  = 16
        self.num_workers = 8


def getConfig(mode: str) -> Config:
    if mode == "train":
        return ConfigTrain()
    if mode == "eval":
        return ConfigEval()
    raise ValueError("mode must be 'train' or 'eval'")
