from typing import List, Tuple


class Config:
    def __init__(self) -> None:
        # =============================== model ============================== #

        self.dim  : int = 3
        self.feats: List[int] = [1, 16, 32]
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

        ## RawDataset
        # read option
        self.threshold: float = 0.0
        # subframe index
        self.h_range: List[int] = [0, 15]
        self.w_range: List[int] = [0, 15]
        # data path
        self.frames_load_folder = "D:/frames"
        self.mlists_load_folder = "D:/mlists"

    def train(self) -> None:
        ## getDataLoader
        self.num: List[int] = [10000, 5000]
        self.type_data: List[str] = ["Sim", "Raw"]
        self.batch_size : int = 1
        self.num_workers: int = 1

        ## Train
        # train
        self.device: str = "cuda"
        self.max_epoch   : int = 200
        self.accumu_steps: int = 10
        # learning rate
        self.lr   : float = 1e-4    # initial learning rate (lr)
        self.gamma: float = 0.95    # decay rate of lr
        # checkpoint
        self.ckpt_save_folder: str  = "ckpt/default"
        self.ckpt_load_path  : str  = ""        # path without .ckpt
        self.ckpt_load_lr    : bool = False     # load lr from ckpt

    def eval(self) -> None:
        ## RawDataset
        self.h_range = [ 9, 12]
        self.w_range = [11, 14]

        ## getDataLoader
        self.num: List[int] = [45000 * 16]
        self.type_data: List[str] = ["Raw"]
        self.batch_size : int = 8
        self.num_workers: int = 4

        ## Eval - also use some config of train
        self.device: str = "cuda"
        self.ckpt_load_path  : str = ""         # path without .ckpt
        self.data_save_folder: str = "data/default"


"""
[4, 4, 4] [1, 32, 64]

For 1&2, we use the 3D UNet without residual and CBAM and the upsample rate
is [4, 4, 4], i.e., the pixel size is 32.5 nm in XYZ. 

For 1, we train the model using learning rate 1e-4 but the result goes to all
dark where the turning point is after 8 epoch. So we load the ckpt 8 and retrain
the 3D UNet with lr at 1e-5.

The result all match the ground truth except there are checkboard artifacts.
They appear in every direction and happean in period of 9 pixels, i.e., 1 pixel
light, 7 pixels dark, 1 pixel light. After check the raw data we use to predict,
these checkbox already exist in the raw data (low resolution) where the period
is 1. These checkbox is very dark but follow the Gaussian distribution, i.e., 
a very small Gaussian point with std around 1.7. So we can not solve this
problem by simply limit the std or lum of our simulation in some range. 

Trainable parameters: 279,969
Training   speed: 1.33s/steps
Validation speed:     s/steps
Evaluation speed:     s/steps
"""


class Config_01(Config):
    def __init__(self) -> None:
        super().__init__()
        self.feats = [1, 32, 64]

    def train(self) -> None:
        super().train()
        ## Train
        self.ckpt_save_folder = "ckpt/01"

    def eval(self) -> None:
        super().eval()
        ## getDataLoader
        self.num = [1000 * 16]
        self.batch_size  = 4
        self.num_workers = 2

        ## Eval
        self.ckpt_load_path   = "ckpt/01/8"
        self.data_save_folder = "data/01"


class Config_02(Config):
    def __init__(self) -> None:
        super().__init__()
        self.feats = [1, 32, 64]

    def train(self) -> None:
        super().train()
        ## Train
        self.lr = 1e-5
        self.ckpt_save_folder = "ckpt/02"
        self.ckpt_load_path   = "ckpt/01/8"

    def eval(self) -> None:
        super().eval()
        ## getDataLoader
        self.batch_size  = 4
        self.num_workers = 2
        
        ## Eval
        self.ckpt_load_path   = "ckpt/02/140"
        self.data_save_folder = "data/02"


"""
[4, 4, 4] [1, 16, 32]

Before move to [4, 8, 8] upsampling rate, we want to try the 3D UNet with less
number of features, i.e., 25% of 1&2 used, on [4, 4, 4] so that the [4, 8, 8]
network can fit in the GPU memory. 

Since 1&2 get the prediction successfully, we may expect to use same training
step in 3&4 except the new number of features. However, from the running log of
3, the learning rate is too large according to previous experiment. Thus, in 4,
we reduce the learning rate to 5e-5. In fact, we did not use same training steps
in 4 as 1&2, i.e., split the training into two part with different learning
rate. The result of 4 is compareable to 1&2 except the checkboard artifacts
become more serious.

Trainable parameters: 70,353
Training   speed: 0.82s/steps
Validation speed: 0.49s/steps
Evaluation speed: 0.28s/steps
"""


class Config_03(Config):
    def train(self) -> None:
        super().train()
        ## Train
        self.ckpt_save_folder = "ckpt/03"

    def eval(self) -> None: raise NotImplementedError


class Config_04(Config):
    def train(self) -> None:
        super().train()
        ## Train
        self.lr = 5e-5
        self.ckpt_save_folder = "ckpt/04"

    def eval(self) -> None:
        super().eval()
        ## Eval
        self.ckpt_load_path   = "ckpt/04/140"
        self.data_save_folder = "data/04"


"""
[4, 4, 4] [1, 16, 32]

For checkbox problem, we may have two possible solution:

In 5, we continue to train the 3D UNet with the raw data. Since there are huge 
number of checkbox, the training process will forcus on the checkbox problem
first. However, after we train to checkpoint 150, the result show that the
network become super unstable and the prediction is similar to the result of
checkpoint 1, which means that the network the network train from the start 
point.

Trainable parameters: 70,353
Training   speed: 0.82s/steps
Validation speed: 0.49s/steps
Evaluation speed: 0.28s/steps
"""


class Config_05(Config):
    def train(self) -> None:
        super().train()
        ## getDataLoader
        self.type_data = ["Raw", "Raw"]

        ## Train
        self.ckpt_save_folder = "ckpt/05"
        self.ckpt_load_path   = "ckpt/04/140"
        self.ckpt_load_lr     = True

    def eval(self) -> None:
        super().eval()
        ## Eval
        self.ckpt_load_path   = "ckpt/05/150"
        self.data_save_folder = "data/05"


class _Config_threshold(Config):
    def train(self) -> None: raise NotImplementedError

    def eval(self) -> None:
        super().eval()
        self.ckpt_load_path   = "ckpt/04/140"


class Config_06(_Config_threshold):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.005
        self.data_save_folder = "data/06"


class Config_07(_Config_threshold):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.010
        self.data_save_folder = "data/07"


class Config_08(_Config_threshold):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.015
        self.data_save_folder = "data/08"


class Config_09(_Config_threshold):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.020
        self.data_save_folder = "data/09"


class Config_10(_Config_threshold):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.025
        self.data_save_folder = "data/10"


class Config_11(_Config_threshold):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.030
        self.data_save_folder = "data/11"


class Config_12(_Config_threshold):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.035
        self.data_save_folder = "data/12"


class Config_13(_Config_threshold):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.040
        self.data_save_folder = "data/13"


def getConfig() -> Tuple[Config, ...]:
    return (
        Config_06(), Config_07(), Config_08(), Config_09(), 
        Config_10(), Config_11(), Config_12(), Config_13()
    )
