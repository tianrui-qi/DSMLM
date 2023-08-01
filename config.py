from typing import List


class Config:
    def __init__(self) -> None:
        # =============================== model ============================== #

        self.dim  : int = 3
        self.feats: List[int] = [1, 32, 64]
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
        self.batch_size : int = 1
        self.num_workers: int = 1

        # =========================== train, eval ============================ #

        ## Train
        # train
        self.device: str = "cuda"
        self.max_epoch   : int = 200
        self.accumu_steps: int = 10
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


"""
[4, 4, 4] [1, 32, 64]

For 2 and 3, we use the 3D UNet without residual and CBAM and the upsample rate
is [4, 4, 4], i.e., the pixel size is 32.5 nm in XYZ. 

For 2, we train the model using learning rate 1e-4 but the result goes to all
dark where the turning point is after 8 epoch. So we load the ckpt 8 and retrain
the 3D UNet with lr at 1e-5.

The result all match the ground truth except there are checkboard artifacts.
They appear in every direction and happean in period of 9 pixels, i.e., 1 pixel
light, 7 pixels dark, 1 pixel light. After check the raw data we use to predict,
these checkbox already exist in the raw data (low resolution) where the period
is 1. These checkbox is very dark but follow the Gaussian distribution, i.e., 
a very small Gaussian point with std around 1.7. So we can not solve this
problem by simply limit the std or lum of our simulation in some range. 

The training speed is 1.33s/steps where the steps size is 10 frames.
"""


class ConfigTrain_2(Config):
    def __init__(self) -> None:
        super().__init__()
        self.ckpt_save_folder = "ckpt/2"


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
        self.lr = 1e-5
        self.ckpt_save_folder = "ckpt/3"
        self.ckpt_load_path = "ckpt/2/8"


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


"""
[4, 8, 8] [1, 16, 32]

Then we move on to [4, 8, 8] upsampling where the final pixel size is 16.25 nm
for XY and 32.5 nm for Z. After some test, we find that the model can train on
4080 when features number decrease to [1, 16, 32], which is 1/2 of the features
number we use in 2&3. However, in 4&5 with lr 1e-4 and 5e-5, the model can not
converge correctly. Thus, we back to [4, 4, 4] upsampling with features 
[1, 16, 32] to see how the model perform. 

Trainable parameters: 70,353            Training   speed: 3.35s/steps
Dedicated GPU memory: 13.9/16.0 GB      Validation speed:
Shared    GPU memory:  0.7/31.9 GB      Evaluation speed: 1.6s/frame

The training speed 250% slower than 2&3. The bottle neck of evalution is cpu 
since when we increase the batch size and number workers from 1 to 2, the speed 
increase from 3.2s/frame to 1.6s/frame.
"""


class ConfigTrain_4(Config):
    def __init__(self) -> None:
        super().__init__()
        self.feats = [1, 16, 32]
        self.up_sample = [4, 8, 8]
        self.ckpt_save_folder = "ckpt/4"


class ConfigEval_4(ConfigTrain_4):
    def __init__(self) -> None:
        super().__init__()
        # data
        self.h_range = [ 9, 12]
        self.w_range = [11, 14]
        self.num = [1000 * 16]
        self.type_data = ["Raw"]
        self.batch_size  = 2
        self.num_workers = 2
        # eval
        checkpoint = 16
        self.ckpt_load_path = "{}/{}".format(self.ckpt_save_folder, checkpoint)
        self.outputs_save_path = "data/4/outputs_{}".format(checkpoint)
        self.labels_save_path  = "data/4/labels_{}".format(checkpoint) 


class ConfigTrain_5(ConfigTrain_4):
    def __init__(self) -> None:
        super().__init__()
        self.lr = 5e-5
        self.ckpt_save_folder = "ckpt/5"


"""
[4, 4, 4] [1, 16, 32]

Trainable parameters: 70,353            Training   speed: 1.22steps/s
Dedicated GPU memory: 6.5/16.0 GB       Validation speed: 2.05steps/s
Shared    GPU memory: 0.4/31.9 GB       Evaluation speed: 3.70steps/s
"""


class ConfigTrain_6(Config):
    def __init__(self) -> None:
        super().__init__()
        self.feats = [1, 16, 32]
        self.ckpt_save_folder = "ckpt/6"


class ConfigTrain_7(Config):
    def __init__(self) -> None:
        super().__init__()
        self.feats = [1, 16, 32]
        self.lr = 5e-5
        self.ckpt_save_folder = "ckpt/7"
        self.ckpt_load_path = "ckpt/7/79"
        self.ckpt_load_lr = True


class ConfigEval_7(ConfigTrain_7):
    def __init__(self) -> None:
        super().__init__()
        # data
        self.h_range = [ 9, 12]
        self.w_range = [11, 14]
        self.num = [45000 * 16]
        self.type_data = ["Raw"]
        self.batch_size  = 8
        self.num_workers = 4
        # eval
        checkpoint = 140
        self.ckpt_load_path = "{}/{}".format(self.ckpt_save_folder, checkpoint)
        self.outputs_save_path = "data/7/outputs_{}".format(checkpoint)
        self.labels_save_path  = "data/7/labels_{}".format(checkpoint) 


def getConfig(mode: str) -> Config:
    if mode == "train":
        return ConfigTrain_7()
    if mode == "eval":
        return ConfigEval_7()
    raise ValueError("mode must be 'train' or 'eval'")
