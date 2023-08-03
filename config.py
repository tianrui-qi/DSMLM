from typing import List


# TODO: split getDataLoader config from file data.py to train.py and eval.py
# TODO: change save path to save folder in eval.py


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

        ## RawDataset
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
        self.ckpt_save_folder: str  = "ckpt"    # folder store ckpt every epoch
        self.ckpt_load_path  : str  = ""        # path without .ckpt
        self.ckpt_load_lr    : bool = False     # load lr from ckpt

    def eval(self) -> None:
        ## getDataLoader
        self.num: List[int] = [45000 * 256]
        self.type_data: List[str] = ["Raw"]
        self.batch_size : int = 4
        self.num_workers: int = 2

        ## Eval - also use some config of train
        self.device: str = "cuda"
        self.ckpt_load_path  : str  = ""        # path without .ckpt
        self.outputs_save_path: str = "data/outputs"    # path without .tif
        self.labels_save_path : str = "data/labels"     # path without .tif


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

Trainable parameters: 279,969           Training   speed: 1.33s/steps
Dedicated GPU memory:   . /16.0 GB      Validation speed:     s/steps
Shared    GPU memory:   . /31.9 GB      Evaluation speed:     s/steps
"""


class Config_1(Config):
    def train(self) -> None:
        super().train()
        ## Train
        self.ckpt_save_folder = "ckpt/1"

    def eval(self) -> None:
        super().eval()
        ## RawDataset
        self.h_range = [ 9, 12]
        self.w_range = [11, 14]
        
        ## getDataLoader
        self.num = [1000 * 16]

        ## Eval
        checkpoint = 8
        self.ckpt_load_path    = "ckpt/1/{}".format(checkpoint)
        self.outputs_save_path = "data/1/outputs_{}".format(checkpoint)
        self.labels_save_path  = "data/1/labels_{}".format(checkpoint) 


class Config_2(Config_1):
    def train(self) -> None:
        super().train()
        ## Train
        self.lr = 1e-5
        self.ckpt_save_folder = "ckpt/2"
        self.ckpt_load_path   = "ckpt/1/8"

    def eval(self) -> None:
        super().eval()
        ## getDataLoader
        self.num = [45000 * 16]
        
        ## Eval
        checkpoint = 140
        self.ckpt_load_path    = "ckpt/2/{}".format(checkpoint)
        self.outputs_save_path = "data/2/outputs_{}".format(checkpoint)
        self.labels_save_path  = "data/2/labels_{}".format(checkpoint) 


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

Trainable parameters: 70,353            Training   speed: 0.82s/steps
Dedicated GPU memory:  6.5/16.0 GB      Validation speed: 0.49s/steps
Shared    GPU memory:  0.4/31.9 GB      Evaluation speed: 0.28s/steps
"""


class Config_3(Config_1):
    def __init__(self) -> None:
        super().__init__()
        self.feats = [1, 16, 32]

    def train(self) -> None:
        super().train()
        ## Train
        self.ckpt_save_folder = "ckpt/3"

    def eval(self) -> None:
        raise NotImplementedError


class Config_4(Config_3):
    def train(self) -> None:
        super().train()
        ## Train
        self.lr = 5e-5
        self.ckpt_save_folder = "ckpt/4"

    def eval(self) -> None:
        super().eval()
        ## getDataLoader
        self.num = [45000 * 16]
        self.batch_size  = 8
        self.num_workers = 4

        ## Eval
        checkpoint = 140
        self.ckpt_load_path    = "ckpt/4/{}".format(checkpoint)
        self.outputs_save_path = "data/4/outputs_{}".format(checkpoint)
        self.labels_save_path  = "data/4/labels_{}".format(checkpoint) 


"""
[4, 4, 4] [1, 16, 32]

One of the possible solution to solve the checkbox problem is to continue to
train the 3D UNet with the raw data. Since there are huge number of checkbox,
the training process will forcus on the checkbox problem first.

However, after we train to checkpoint 150, the result show that the network
become super unstable and the prediction is similar to the result of checkpoint
1, which means that the network the network train from the start point.

Trainable parameters: 70,353            Training   speed: 0.82s/steps
Dedicated GPU memory:  6.5/16.0 GB      Validation speed: 0.49s/steps
Shared    GPU memory:  0.4/31.9 GB      Evaluation speed: 0.28s/steps
"""


class Config_5(Config_4):
    def train(self) -> None:
        super().train()
        ## getDataLoader
        self.type_data = ["Raw", "Raw"]

        ## Train
        self.ckpt_save_folder = "ckpt/5"
        self.ckpt_load_path = "ckpt/4/140"
        self.ckpt_load_lr = True

    def eval(self) -> None:
        super().eval()
        ## Eval
        checkpoint = 150
        self.ckpt_load_path    = "ckpt/5/{}".format(checkpoint)
        self.outputs_save_path = "data/5/outputs_{}".format(checkpoint)
        self.labels_save_path  = "data/5/labels_{}".format(checkpoint) 


def getConfig() -> Config: return Config_5()
