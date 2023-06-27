from typing import List


class Config:
    def __init__(self):
        # dimensional config
        # MUST be same accross whole pipline
        self.dim_frame: List[int] = [64, 64, 64]    # [C, H, W], by pixel
        self.up_sample: List[int] = [ 4,  4,  4]    # [C, H, W], by scale

        # ============================= train.py ============================= #

        ## (class) Train
        # train
        self.max_epoch: int = 400
        # learning rate
        self.lr   : float = 0.00001     # initial learning rate (lr)
        self.gamma: float = 0.95
        # checkpoint
        self.cpt_save_path : str  = "checkpoints"   # path without .pt
        self.cpt_save_epoch: bool = False           # save pt every epoch
        self.cpt_load_path : str  = ""              # path without .pt
        self.cpt_load_lr   : bool = False           # load lr from cpt

        # ============================= model.py ============================= #

        ## (class) Criterion
        self.kernel_size : int   = 7    # kernel size of GaussianBlur
        self.kernel_sigma: float = 1.0  # sigma of kernel
        self.l1_coeff    : float = 0.0  # set 1 to repeat deep storm

        # ============================= data.py ============================== #

        ## (def) getData
        self.num : List[int] = [3000  , 900   ]     # num of train, valid data
        self.type: List[str] = ["Simu", "Simu"]     # type of train, valid data

        ## (class) SimDataLoader, CropDataLoader
        self.batch_size : int = 3
        self.num_workers: int = 3

        ## (class) SimDataset
        # config for adjust distribution of molecular
        self.mol_epoch: int         = 128           # num mol simulated / epoch
        self.mol_range: List[int]   = [0, 64]       # min, max num of mol/frame
        self.std_range: List[float] = [0.5, 2.5]    # by pixel in low resolution
        self.lum_range: List[float] = [1/32, 1.0]
        # config for reducing resolution and adding noise
        self.bitdepth : int   = 8
        self.qe       : float = 0.82
        self.sen      : float = 5.88
        self.noise_mu : float = 0.0    # mu of Gaussian noise, by 2^bitdepth
        self.noise_std: float = 0.0    # variance of dark noise, by 2^bitdepth

        ## (class) RawDataset
        self.raw_folder  = "data/raw"


class Test_8(Config):
    def __init__(self):
        super().__init__()
        # train
        self.cpt_save_path  = "checkpoints/test_8"
        self.cpt_save_epoch = True


class Test_9(Config):
    def __init__(self):
        super().__init__()
        # train
        self.lr = 0.0001
        self.cpt_save_path  = "checkpoints/test_9"
        self.cpt_save_epoch = True
        self.cpt_load_path  = "checkpoints/test_8"
        # data
        self.type = ["Simu", "Crop"]
        self.noise_mu  = 8.0
        self.noise_std = 8.0
