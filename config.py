from typing import List


class ConfigModel:
    def __init__(self) -> None:
        self.dim  : int = 2
        self.feats: List[int] = [80, 1280, 2560]
        self.use_res : bool = True
        self.use_cbam: bool = True
        self.use_att : bool = True 


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
        self.up_sample: List[int] = [ 2,  4,  4]    # [C, H, W], by scale

        ## SimDataset
        # config for adjust distribution of molecular
        self.mol_range: List[int] = [0, 96]    # min, max num of mol/frame
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
        self.num : List[int] = [7680 , 2560 ]
        self.type: List[str] = ["Sim", "Raw"]
        self.batch_size : int = 2
        self.num_workers: int = 2

        # =========================== train, eval ============================ #

        ## Train
        # train
        self.device: str = "cuda"
        self.max_epoch   : int = 400
        self.accumu_steps: int = 16
        # learning rate
        self.lr   : float = 1e-4    # initial learning rate (lr)
        self.gamma: float = 0.96    # decay rate of lr
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
        self.h_range = [0, 15]
        self.w_range = [8, 13]
        ## getDataLoader
        self.num  = [25 * 96]
        self.type = ["Raw"]
        self.batch_size  = 4
        self.num_workers = 2
        ## Eval
        checkpoint = 1
        self.ckpt_load_path = "{}/{}".format(self.ckpt_save_folder, checkpoint)
        self.outputs_save_path = "data/outputs_{}".format(checkpoint)
        self.labels_save_path  = "data/labels_{}".format(checkpoint) 


def getConfig(mode: str) -> Config:
    if mode == "train":
        return ConfigTrain()
    if mode == "eval":
        return ConfigEval()
    raise ValueError("mode must be 'train' or 'eval'")
