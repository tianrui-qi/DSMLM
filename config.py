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
        self.max_epoch: int = 1000
        self.accumulation_steps: int = 1    # unit: batch
        # learning rate
        self.lr   : float = 1e-5     # initial learning rate (lr)
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

        ## (def) getDataLoader
        self.num : List[int] = [3000 , 900  ]  # num of train, valid data
        self.type: List[str] = ["Sim", "Sim"]  # type of train, valid data
        self.batch_size : int = 3
        self.num_workers: int = 3

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


class Test_1(Config):
    def __init__(self):
        super().__init__()
        # train
        self.cpt_save_path  = "checkpoints/test_1"
        self.cpt_save_epoch = True
        # data
        self.mol_range = [0, 64]
        self.lum_range = [1/8, 3/4]
        self.type = ["Sim", "Raw"]


class Test_2(Config):
    def __init__(self):
        super().__init__()
        # train
        self.lr             = 1e-6
        self.cpt_save_path  = "checkpoints/test_2"
        self.cpt_save_epoch = True
        self.cpt_load_path  = "checkpoints/test_1"
        # data
        self.mol_range = [0, 64]
        self.lum_range = [1/8, 3/4]
        self.type = ["Sim", "Raw"]


class Test_3(Config):
    def __init__(self):
        super().__init__()
        # train
        self.lr             = 1e-7
        self.cpt_save_path  = "checkpoints/test_3"
        self.cpt_save_epoch = True
        self.cpt_load_path  = "checkpoints/test_2/184"
        # data
        self.mol_range = [0, 64]
        self.lum_range = [1/8, 3/4]
        self.type = ["Sim", "Raw"]


class Test_4(Config):
    def __init__(self):
        super().__init__()
        # train
        self.gamma          = 0.99
        self.cpt_save_path  = "checkpoints/test_4"
        self.cpt_save_epoch = True
        self.cpt_load_path  = "checkpoints/test_3/400"
        # data
        self.type = ["Sim", "Raw"]


class Test_5(Config):
    def __init__(self):
        super().__init__()
        # train
        self.gamma          = 0.99
        self.cpt_save_path  = "checkpoints/test_5"
        self.cpt_save_epoch = True
        self.cpt_load_path  = "checkpoints/test_4/403"
        self.cpt_load_lr    = True
        # data
        self.type = ["Sim", "Raw"]


class Test_6(Config):
    def __init__(self):
        super().__init__()
        # train
        self.lr             = 1e-5
        self.gamma          = 0.99
        self.cpt_save_path  = "checkpoints/test_6"
        self.cpt_save_epoch = True
        self.cpt_load_path  = "checkpoints/test_2/184"
        # data
        self.lum_range = [1/8, 1.0]
        self.type = ["Sim", "Raw"]


class Test_7(Config):
    def __init__(self):
        super().__init__()
        # train
        self.accumulation_steps = 4
        self.lr = 1e-4
        self.cpt_save_path  = "checkpoints/test_7"
        self.cpt_save_epoch = True
        # data
        self.num  = [8000 , 2000 ]
        self.type = ["Sim", "Raw"]
        self.batch_size  = 2
        self.num_workers = 2
