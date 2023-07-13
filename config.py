from typing import List


class Config:
    def __init__(self) -> None:
        # dimensional config
        # MUST be same accross whole pipline
        self.dim_frame: List[int] = [64, 64, 64]    # [C, H, W], by pixel
        self.up_sample: List[int] = [ 2,  8,  8]    # [C, H, W], by scale

        # =============================== train ============================== #

        ## (Class) Train
        # train
        self.device: str = "cuda"
        self.max_epoch: int = 200
        self.accumulation_steps: int = 8    # unit: batch
        # learning rate
        self.lr   : float = 1e-3     # initial learning rate (lr)
        self.gamma: float = 0.95
        # checkpoint
        self.ckpt_save_path : str  = "ckpt/train"   # path without .pt
        self.ckpt_save_epoch: bool = True           # save pt every epoch
        self.ckpt_load_path : str  = ""             # path without .pt
        self.ckpt_load_lr   : bool = False          # load lr from ckpt

        # =============================== eval =============================== #

        ## (class) Eval - also use some config of train and data
        # train: self.device, self.ckpt_load_path
        # eval
        self.result_save_path: str = "data/eval"    # path without .tif
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
            [1.6, 1.3, 1.3],  # [500, 400, 400] nm for FWHM, pixel size 65nm
            [3.0, 1.6, 1.6],  # [900, 500, 500] nm for FWHM, pixel size 65nm
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
        self.num : List[int] = [8000 , 2000 ]   # num of train, valid data
        self.type: List[str] = ["Sim", "Sim"]   # Sim or Raw
        self.batch_size : int = 2
        self.num_workers: int = 2


class ConfigTrain_1(Config):
    def __init__(self) -> None:
        super().__init__()
        ## (Class) Train
        self.ckpt_save_path  = "ckpt/train_1"
        ## (def) getDataLoader
        self.type = ["Sim", "Raw"]


class ConfigEval_1(Config):
    def __init__(self) -> None:
        super().__init__()
        ## (class) Eval
        self.ckpt_load_path   = "ckpt/train_1"
        self.result_save_path = "data/eval/result"
        ## (class) RawDataset
        self.h_range = [3, 8]
        self.w_range = [4, 9]
        self.num_sub = 36
        ## (def) getDataLoader
        self.num  = [30 * self.num_sub]
        self.type = ["Raw"]
        self.batch_size = 12


def getConfig(mode: str) -> Config:
    if mode == "train":
        return ConfigTrain_1()
    if mode == "eval":
        return ConfigEval_1()
    raise ValueError("mode must be 'train' or 'eval'")
