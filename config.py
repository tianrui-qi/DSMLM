from typing import List


class Config:
    def __init__(self) -> None:
        # =============================== model ============================== #

        self.dim  : int = 2
        self.feats: List[int] = [160, 320, 640]
        self.use_res : bool = False
        self.use_cbam: bool = False

        # =============================== loss =============================== #

        self.type_loss   : str   = "l2"     # l1, l2
        self.kernel_size : int   = 7
        self.kernel_sigma: float = 1.0

        # =============================== data =============================== #

        # dimensional config - MUST be same accross whole pipline
        self.dim_frame: List[int] = [40, 40, 40]    # [C, H, W], by pixel
        self.up_sample: List[int] = [ 4,  4,  4]    # [C, H, W], by scale

        ## SimDataset
        # config for adjust distribution of molecular
        self.mol_range: List[int] = [0, 128]    # min, max num of mol/frame
        self.std_range: List[List[float]] = [   # std range of each dimension
            [1.0, 1.0, 1.0],  # FWHM [300 300 300], pixel size [130 130 130]
            [2.5, 2.0, 2.0],  # FWHM [800 600 600], pixel size [130 130 130]
        ]
        self.lum_range: List[float] = [0.0, 1.0]
        # config for reducing resolution and adding noise
        self.bitdepth   : int   = 16
        self.qe         : float = 0.82
        self.sensitivity: float = 5.88
        self.dark_noise : float = 2.29

        ## RawDataset
        # subframe index
        self.h_range: List[int] = [0, 15]
        self.w_range: List[int] = [0, 15]
        # data path
        self.frames_load_folder = "D:/frames"
        self.mlists_load_folder = "D:/mlists"

        ## getDataLoader
        self.num: List[int] = [10000, 5000]
        self.type_data: List[str] = ["Sim", "Raw"]
        self.batch_size : int = 2
        self.num_workers: int = 2

        # =========================== train, eval ============================ #

        ## Train
        # train
        self.device: str = "cuda"
        self.max_epoch   : int = 200
        self.accumu_steps: int = 5
        # learning rate
        self.lr   : float = 1e-4    # initial learning rate (lr)
        self.gamma: float = 0.95    # decay rate of lr
        # checkpoint
        self.ckpt_save_folder: str  = "ckpt"    # folder store ckpt every epoch
        self.ckpt_load_path  : str  = ""        # path without .ckpt
        self.ckpt_load_lr    : bool = False     # load lr from ckpt

        ## Eval - also use some config of train
        self.outputs_save_path: str = "data/outputs"    # path without .tif
        self.labels_save_path : str = "data/labels"     # path without .tif


class ConfigTrain_1(Config):
    def __init__(self) -> None:
        super().__init__()
        self.dim = 2
        self.feats = [160, 320, 640]
        self.use_res  = True
        self.use_cbam = True
        self.ckpt_save_folder = "ckpt/1"


class ConfigEval_1(ConfigTrain_1):
    def __init__(self) -> None:
        super().__init__()
        # data
        self.h_range = [ 9, 12]
        self.w_range = [11, 14]
        self.num = [1000 * 16]
        self.type_data = ["Raw"]
        self.batch_size  = 8
        self.num_workers = 2
        # eval
        checkpoint = 0
        self.ckpt_load_path = "{}/{}".format(self.ckpt_save_folder, checkpoint)
        self.outputs_save_path = "data/1/outputs_{}".format(checkpoint)
        self.labels_save_path  = "data/1/labels_{}".format(checkpoint) 


class ConfigTrain_2(Config):
    def __init__(self) -> None:
        super().__init__()
        self.dim = 3
        self.feats = [1, 32, 64]
        self.ckpt_save_folder = "ckpt/2"
        self.batch_size = 1
        self.num_workers = 1
        self.accumu_steps = 10


class ConfigEval_2(ConfigTrain_2):
    def __init__(self) -> None:
        super().__init__()
        # data
        self.h_range = [ 9, 12]
        self.w_range = [11, 14]
        self.num = [1000 * 16]
        self.type_data = ["Raw"]
        self.batch_size  = 4
        self.num_workers = 2
        # eval
        checkpoint = 8
        self.ckpt_load_path = "{}/{}".format(self.ckpt_save_folder, checkpoint)
        self.outputs_save_path = "data/2/outputs_{}".format(checkpoint)
        self.labels_save_path  = "data/2/labels_{}".format(checkpoint) 


class ConfigTrain_3(ConfigTrain_2):
    def __init__(self) -> None:
        super().__init__()
        self.ckpt_save_folder = "ckpt/3"
        self.ckpt_load_path = "ckpt/2/8"
        self.lr = 1e-5


class ConfigEval_3(ConfigTrain_3):
    def __init__(self) -> None:
        super().__init__()
        # data
        self.h_range = [ 9, 12]
        self.w_range = [11, 14]
        self.num = [45000 * 16]
        self.type_data = ["Raw"]
        self.batch_size  = 4
        self.num_workers = 2
        # eval
        checkpoint = 140
        self.ckpt_load_path = "{}/{}".format(self.ckpt_save_folder, checkpoint)
        self.outputs_save_path = "data/3/outputs_{}".format(checkpoint)
        self.labels_save_path  = "data/3/labels_{}".format(checkpoint) 


def getConfig(mode: str) -> Config:
    if mode == "train":
        return ConfigTrain_3()
    if mode == "eval":
        return ConfigEval_3()
    raise ValueError("mode must be 'train' or 'eval'")
