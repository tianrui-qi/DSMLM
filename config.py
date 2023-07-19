from typing import List


class ConfigModel:
    def __init__(self) -> None:
        self.dim  : int = 3
        self.feats: List[int] = [1, 32, 64]
        self.residual : bool = True
        self.attention: bool = True 


class ConfigLoss:
    def __init__(self) -> None:
        self.type: str = "l2"  # l1, l2
        self.kernel_size : int   = 7
        self.kernel_sigma: float = 1.0


class Config:
    def __init__(self) -> None:
        self.config_model = ConfigModel()
        self.config_loss  = ConfigLoss()

        # =============================== data =============================== #

        # dimensional config - MUST be same accross whole pipline
        self.dim_frame: List[int] = [40, 40, 40]    # [C, H, W], by pixel
        self.up_sample: List[int] = [ 4,  4,  4]    # [C, H, W], by scale

        ## SimDataset
        # config for adjust distribution of molecular
        self.mol_range: List[int] = [0, 128]    # min, max num of mol/frame
        self.std_range: List[List[float]] = [   # std range of each dimension
            [2.0, 1.0, 1.0],  # FWHM [300 300 300], pixel size [65 130 130]
            [5.0, 2.0, 2.0],  # FWHM [800 600 600], pixel size [65 130 130]
        ]
        self.lum_range: List[float] = [0.0, 1.0]
        # config for reducing resolution and adding noise
        self.bitdepth   : int   = 16
        self.qe         : float = 0.82
        self.sensitivity: float = 5.88
        self.dark_noise : float = 2.29

        ## RawDataset
        # subframe index
        self.h_range: List[int] = [0, 9]
        self.w_range: List[int] = [0, 9]
        # data path
        self.frames_load_folder = "D:/frames"
        self.mlists_load_folder = "D:/mlists"

        ## getDataLoader
        self.num : List[int] = [10000, 5000 ]
        self.type_data: List[str] = ["Sim", "Raw"]
        self.batch_size : int = 1       # [1, 2, 4, 5, 10, 20, 25, 50]
        self.num_workers: int = 1

        # =========================== train, eval ============================ #

        ## Train
        # train
        self.device: str = "cuda"
        self.max_epoch   : int = 400
        self.accumu_steps: int = 50     # [100, 50, 25, 20, 10, 5, 4, 2]
        # learning rate
        self.lr   : float = 1e-3        # initial learning rate (lr)
        self.gamma: float = 0.96        # decay rate of lr
        # checkpoint
        self.ckpt_save_folder: str  = "ckpt"    # folder store ckpt every epoch
        self.ckpt_load_path  : str  = ""        # path without .ckpt
        self.ckpt_load_lr    : bool = False     # load lr from ckpt

        ## Eval - also use some config of train and data
        self.outputs_save_path: str = "data/outputs"    # path without .tif
        self.labels_save_path : str = "data/labels"     # path without .tif


class ConfigTrain(Config):
    def __init__(self) -> None:
        super().__init__()


class ConfigEval(ConfigTrain):
    def __init__(self) -> None:
        super().__init__()
        ## RawDataset
        self.h_range = [4, 7]
        self.w_range = [6, 9]
        ## getDataLoader
        self.num  = [5000 * 16]
        self.type_data = ["Raw"]
        self.batch_size  = 8
        self.num_workers = 4
        ## Eval
        checkpoint = 6
        self.ckpt_load_path = "{}/{}".format(self.ckpt_save_folder, checkpoint)
        self.outputs_save_path = "data/outputs_{}".format(checkpoint)
        self.labels_save_path  = "data/labels_{}".format(checkpoint) 


def getConfig(mode: str) -> Config:
    if mode == "train":
        return ConfigTrain()
    if mode == "eval":
        return ConfigEval()
    raise ValueError("mode must be 'train' or 'eval'")
