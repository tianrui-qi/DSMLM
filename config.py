from typing import List


class Config:
    def __init__(self):
        # dimensional config
        # MUST be same accross whole pipline
        self.dim_frame: List[int] = [64, 64, 64]    # [C, H, W], by pixel
        self.up_sample: List[int] = [ 4,  4,  4]    # [C, H, W], by scale

        # =========================== train & eval =========================== #

        ## (class) Train
        # train
        self.max_epoch: int = 1000
        self.accumulation_steps: int = 4    # unit: batch
        # learning rate
        self.lr   : float = 1e-4     # initial learning rate (lr)
        self.gamma: float = 0.95
        # checkpoint
        self.cpt_save_path : str  = "checkpoints"   # path without .pt
        self.cpt_save_epoch: bool = False           # save pt every epoch
        self.cpt_load_path : str  = ""              # path without .pt
        self.cpt_load_lr   : bool = False           # load lr from cpt

        # =============================== model ============================== #

        ## (class) Criterion
        self.kernel_size : int   = 7    # kernel size of GaussianBlur
        self.kernel_sigma: float = 1.0  # sigma of kernel

        # =============================== data =============================== #

        ## (def) getDataLoader
        self.num : List[int] = [8000 , 2000 ]  # num of train, valid data
        self.type: List[str] = ["Sim", "Sim"]  # type of train, valid data
        self.batch_size : int = 2
        self.num_workers: int = 2

        ## (class) SimDataset
        # config for adjust distribution of molecular
        self.mol_range: List[int] = [0, 128]     # min, max num of mol/frame
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
        self.raw_folder  = "data/raw"


class Test_7(Config):
    def __init__(self):
        super().__init__()
        # train
        self.cpt_save_path  = "checkpoints/test_7"
        self.cpt_save_epoch = True
        # data
        self.type = ["Sim", "Raw"]


class Eval(Config):
    def __init__(self):
        super().__init__()
        self.cpt_load_path = "checkpoints/test_7/30"
        self.num  = [1000000]
        self.type = ["Raw"]
        self.batch_size = 10
