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
Evaluation speed:     s/frames
"""


class d_01(Config):
    def __init__(self) -> None:
        super().__init__()
        self.feats = [1, 32, 64]

    def train(self) -> None:
        super().train()
        ## Train
        self.ckpt_save_folder = "ckpt/d-chessboard/01"

    def eval(self) -> None:
        super().eval()
        ## getDataLoader
        self.num = [1000 * 16]
        self.batch_size  = 4
        self.num_workers = 2

        ## Eval
        self.ckpt_load_path   = "ckpt/d-chessboard/01/8"
        self.data_save_folder = "data/d-chessboard/01"


class d_02(Config):
    def __init__(self) -> None:
        super().__init__()
        self.feats = [1, 32, 64]

    def train(self) -> None:
        super().train()
        ## Train
        self.lr = 1e-5
        self.ckpt_save_folder = "ckpt/d-chessboard/02"
        self.ckpt_load_path   = "ckpt/d-chessboard/01/8"

    def eval(self) -> None:
        super().eval()
        ## getDataLoader
        self.batch_size  = 4
        self.num_workers = 2
        
        ## Eval
        self.ckpt_load_path   = "ckpt/d-chessboard/02/140"
        self.data_save_folder = "data/d-chessboard/02"


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
Evaluation speed: 0.28s/frames
"""


class d_03(Config):
    def train(self) -> None:
        super().train()
        ## Train
        self.ckpt_save_folder = "ckpt/d-chessboard/03"

    def eval(self) -> None: raise NotImplementedError


class d_04(Config):
    def train(self) -> None:
        super().train()
        ## Train
        self.lr = 5e-5
        self.ckpt_save_folder = "ckpt/d-chessboard/04"

    def eval(self) -> None:
        super().eval()
        ## Eval
        self.ckpt_load_path   = "ckpt/d-chessboard/04/140"
        self.data_save_folder = "data/d-chessboard/04"


"""
[4, 4, 4] [1, 16, 32]

For checkbox problem, we may have two possible solutions:

In 5, we continue to train the 3D UNet with the raw data. Since there are huge 
number of checkbox, the training process will forcus on the checkbox problem
first. However, after we train to checkpoint 150, the result show that the
network become super unstable and the prediction is similar to the result of
checkpoint 1, which means that the network the network train from the start 
point.

Trainable parameters: 70,353
Training   speed: 0.82s/steps
Validation speed: 0.49s/steps
Evaluation speed: 0.28s/frames
"""


class d_05(Config):
    def train(self) -> None:
        super().train()
        ## getDataLoader
        self.type_data = ["Raw", "Raw"]

        ## Train
        self.ckpt_save_folder = "ckpt/d-chessboard/05"
        self.ckpt_load_path   = "ckpt/d-chessboard/04/140"
        self.ckpt_load_lr     = True

    def eval(self) -> None:
        super().eval()
        ## Eval
        self.ckpt_load_path   = "ckpt/d-chessboard/05/150"
        self.data_save_folder = "data/d-chessboard/05"


"""
[4, 4, 4] [1, 16, 32]

Another possible solution is to add some threshold such that if the pixel value
is less than the threhold, we set it as 0.

In 06 old, we test threshold from 0.0 to 0.1 where the step size is 0.005. 
1.  When the threshold is 0.04, the checkbox in the background, i.e.,
    these checkbox without tube structure will be removed. However, the checkbox
    that combine / mix with the structure still exits. 
2.  As threshold increase, these is a new checkbox appear where the size is 5.
    These checkbox all on the right up side of the tube structure and seem like
    some kind of shift of the tube structures. 
We gauss reason that cause the size 5 checkbox is not same as the size of 9.
Thus, now we have two problem need to solve, and for each checkbox, we have some 
preliminary solution:
9.  For different layer, the luminance of the checkbox may be different. Thus,
    when threshold is 0.04, these checkbox that are very dark been removed where
    these checkbox mix with the tube structure is very bright and need more
    threshold. However, either dark or bright checkbox follow same proportion: 
    they all apppear in same proportion of the whole brightness. Thus, we need
    set different threshold for different layer pixel based on their whole
    brightness.
5.  The gauss reason for size 5 checkbox is that for two nearby pixel, after 
    scale up by four, the network give two prediction to each pixel where the
    distance will be 5. This may cause by overfitting and we may try some
    different checkpoint to see how will the result change. 

Before further investigation, we change the normolization of the raw data from 
each frame's max value to a fixed value 6.5. We get this value by calculate
average maximum value of all frames. Then, we test the new normalization logic
in 06 where threshold change from 0.0 to 0.1 with step size 0.01. Two problem 
discrible above still exist. We will use this new normalization logic in future
test.
"""


class d_06(Config):
    def train(self) -> None: raise NotImplementedError

    def eval(self) -> None:
        super().eval()
        self.ckpt_load_path = "ckpt/d-chessboard/04/140"


class d_06_000(d_06):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.000
        self.data_save_folder = "data/d-chessboard/06/000"


class d_06_010(d_06):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.010
        self.data_save_folder = "data/d-chessboard/06/010"


class d_06_020(d_06):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.020
        self.data_save_folder = "data/d-chessboard/06/020"


class d_06_030(d_06):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.030
        self.data_save_folder = "data/d-chessboard/06/030"


class d_06_040(d_06):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.040
        self.data_save_folder = "data/d-chessboard/06/040"


class d_06_050(d_06):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.050
        self.data_save_folder = "data/d-chessboard/06/050"


class d_06_060(d_06):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.060
        self.data_save_folder = "data/d-chessboard/06/060"


class d_06_070(d_06):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.070
        self.data_save_folder = "data/d-chessboard/06/070"


class d_06_080(d_06):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.080
        self.data_save_folder = "data/d-chessboard/06/080"


class d_06_090(d_06):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.090
        self.data_save_folder = "data/d-chessboard/06/090"


class d_06_100(d_06):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.100
        self.data_save_folder = "data/d-chessboard/06/100"


"""
"""


class d_07(Config):
    def train(self) -> None: raise NotImplementedError

    def eval(self) -> None:
        super().eval()
        self.ckpt_load_path = "ckpt/d-chessboard/04/60"


class d_07_000(d_07):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.000
        self.data_save_folder = "data/d-chessboard/07/000"


class d_07_010(d_07):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.010
        self.data_save_folder = "data/d-chessboard/07/010"


class d_07_020(d_07):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.020
        self.data_save_folder = "data/d-chessboard/07/020"


class d_07_030(d_07):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.030
        self.data_save_folder = "data/d-chessboard/07/030"


class d_07_040(d_07):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.040
        self.data_save_folder = "data/d-chessboard/07/040"


class d_07_050(d_07):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.050
        self.data_save_folder = "data/d-chessboard/07/050"


class d_07_060(d_07):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.060
        self.data_save_folder = "data/d-chessboard/07/060"


class d_07_070(d_07):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.070
        self.data_save_folder = "data/d-chessboard/07/070"


class d_07_080(d_07):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.080
        self.data_save_folder = "data/d-chessboard/07/080"


class d_07_090(d_07):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.090
        self.data_save_folder = "data/d-chessboard/07/090"


class d_07_100(d_07):
    def eval(self) -> None:
        super().eval()
        self.threshold = 0.100
        self.data_save_folder = "data/d-chessboard/07/100"


def getConfig() -> Tuple[Config, ...]: return (
    d_07_000(),
    d_07_010(), d_07_020(), d_07_030(), d_07_040(), d_07_050(),
    d_07_060(), d_07_070(), d_07_080(), d_07_090(), d_07_100(),
)
