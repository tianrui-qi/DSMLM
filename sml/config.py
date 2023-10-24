from typing import List


class Config:
    def __init__(self) -> None:
        ## SimDataset & RawDataset
        # luminance/brightness information
        self.lum_info: bool = True
        # dimension
        self.dim_dst: List[int] = [160, 160, 160]  # [C, H, W], pixel

        ## SimDataset
        # scale up factor
        self.scale_list: List[int] = [2, 4, 8, 16]
        # molecular profile
        self.std_src: List[List[float]] = [   # std range
            [1.0, 1.0, 1.0],  # minimum std, [C, H, W], by pixel
            [3.0, 2.5, 2.5],  # maximum std, [C, H, W], by pixel
        ]

        ## RawDataset
        # scale up factor
        self.scale  : List[int] = [4, 4, 4]     # [C, H, W]
        # pad for patching
        self.pad_src: List[int] = [2, 2, 2]     # [C, H, W], pixel
        # data path
        self.frames_load_fold: str = "D:/hela/frames"
        self.mlists_load_fold: str = "D:/hela/mlists"

        ## ResAttUNet
        self.dim  : int = 3
        self.feats: List[int] = [1, 16, 32]
        self.use_res : bool = False
        self.use_cbam: bool = False

        ## GaussianBlurLoss
        self.kernel_size : int   = 7
        self.kernel_sigma: float = 1.0


class ConfigTrainer(Config):
    def __init__(self) -> None:
        super().__init__()
        self.accumu_steps: int = 10
        # path
        self.ckpt_save_fold: str  = "ckpt/default"
        self.ckpt_load_path: str  = ""        # path without .ckpt
        self.ckpt_load_lr  : bool = False     # load lr from ckpt
        # dataloader
        self.num: List[int] = [10000, 5000]   # train and valid
        self.batch_size : int = 1
        # optimizer
        self.lr   : float = 1e-4    # initial learning rate (lr)
        self.gamma: float = 0.95    # decay rate of lr


class ConfigEvaluator(Config):
    def __init__(self) -> None:
        super().__init__()
        # path
        self.ckpt_load_path: str = ""   # path without .ckpts
        self.data_save_fold: str = "data/default"
        # dataloader
        self.batch_size : int = 5


""" new architecture
To solve the super resolution problem, we complete redesign the training
process. More specifically, now we fixed the dim_dst to [160 160 160] instead
of variante according to the scale up factor. By fix the input of the network,
the size of the network will independent to the scale up factor. The scale up
factor will only affect how we cut the raw data into subframes to feed into the
network. Also, now we can train various scale up factor at the same time.

Result:
In e01, the learning rate is too large, so we reduce it to 1e-5 in e02. In e03,
we evaluate the network with hela dataset with scale up factor [4, 4, 4]. The
result show very obvious artifacts with a len 5. The chessbox is different from
before. 

up sampling rate: [4, 4, 4]
features number : [1, 16, 32]
Trainable paras : 70,353
Training   speed: 1.14 steps /s ( 10 iterations/step)
Validation speed:      steps /s ( 10 iterations/step)
Evaluation speed: 3.63 frames/s ( 16 subframes/frame)
"""


class e03(ConfigEvaluator):
    def __init__(self) -> None:
        super().__init__()
        self.scale = [4, 4, 4]
        self.ckpt_load_path = "ckpt/e02/210" 
        self.data_save_fold = "data/e-sr/03"
        self.batch_size : int = 4


class e02(ConfigTrainer):
    def __init__(self) -> None:
        super().__init__()
        #self.lr = 1e-5
        #self.ckpt_load_path = "ckpt/e01/150"
        self.ckpt_load_path = "ckpt/e02/172"
        self.ckpt_save_fold = "ckpt/e02"
        self.ckpt_load_lr   = True


class e01(ConfigTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.ckpt_load_path = "ckpt/d04/140"
        self.ckpt_save_fold = "ckpt/e01"


# All the following configs are no longer maintained.


""" whole field of view
Next we generate a field of view frames as a summary of section d-chessboard of test; for section e, we will move on to 8 times super resolution.

Result:
In d08, hela cell's result is good. However, a new group of data in d09 show
that the network can not predict them very well. This group of data as 
bitdepth 8 instead 32 in d08. This equivalent to add a threshold to the raw
data and cause a lot of size 5 checkbox, similar to result of d06 after we add
too much threshold. Thus, for futurn training, we have to add random noise for
bitdepth where before we always set it at 16.

up sampling rate: [4, 4, 4]
features number : [1, 16, 32]
Trainable paras : 70,353
Training   speed:      steps /s ( 10 iterations/step)
Validation speed:      steps /s ( 10 iterations/step)
Evaluation speed: 0.24 frames/s (256 subframes/frame)
"""


class d09(Config):
    def __init__(self) -> None:
        super().__init__()
        self.frames_load_fold = "C:/Users/tianrui/Desktop/frames"

    def train(self) -> None: return NotImplementedError

    def eval(self) -> None:
        super().eval()
        ## RawDataset
        self.h_range = [ 0, 15]
        self.w_range = [ 0, 15]

        ## getDataLoader
        self.num: List[int] = [40000 * 256]

        ## Eval
        self.ckpt_load_path = "ckpt/d07/170"
        self.data_save_fold = "data/d-artefact/09"


class d08(Config):
    def train(self) -> None: return NotImplementedError

    def eval(self) -> None:
        super().eval()
        ## RawDataset
        self.h_range = [ 0, 15]
        self.w_range = [ 0, 15]

        ## getDataLoader
        self.num: List[int] = [45000 * 256]

        ## Eval
        self.ckpt_load_path = "ckpt/d07/170"
        self.data_save_fold = "data/d-artefact/08"


""" luminance information
In 07, we implement a new parameters in __init__ of Config class, i.e., 
lum_info, where it can be set to True or False. Before, we always set center
of each Gaussian as 1.0, which means that our label is in fact binary, i.e.,
this label only tell us where is the center of the Gaussian. After we set 
`lum_info` to True, the label generate by Sim&Raw dataset will time the frame
elementwise. Thus, the label will tell us the luminance of each pixel. Then, we
continue to train the 3D UNet with the new label from ckpt 140 of 04.

Result:
The network can predict the luminance of each pixel where the chessboard in dark
area disappear, but still exist in the predicition with high density of 
molecular. We check the output of DeepSTORM and they also have the same problem
in high density area. Thus, current result for test set d-chessboard is good 
enough.

Note:
We remove the self.lum_info from the config since we must use it to solve the
checkbox problem. That is, we logicly set lum_info to True permanently in our
implementation of data.

up sampling rate: [4, 4, 4]
features number : [1, 16, 32]
Trainable paras : 70,353
Training   speed:      steps /s ( 10 iterations/step)
Validation speed:      steps /s ( 10 iterations/step)
Evaluation speed: 3.80 frames/s ( 16 subframes/frame)
"""


class d07(Config):
    def train(self) -> None:
        super().train()
        ## Train
        self.lr = 1e-5
        self.ckpt_load_path = "ckpt/d04/140"
        self.ckpt_save_fold = "ckpt/d07"

    def eval(self) -> None:
        super().eval()
        ## Eval
        self.ckpt_load_path = "ckpt/d07/170"
        self.data_save_fold = "data/d-artefact/07"


""" threshold
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

Result:
Threshold or some maual operation may not be a good solution for cheessboard
since first, we have to set them manually by experiment every time when we have
raw data and how to judge the result is good or not is also a problem. Second,
this kind of operation may cause some other artifacts.

Note:
The config for set threshold has been deleted.

up sampling rate: [4, 4, 4]
features number : [1, 16, 32]
Trainable paras : 70,353
Training   speed:      steps /s ( 10 iterations/step)
Validation speed:      steps /s ( 10 iterations/step)
Evaluation speed: 3.80 frames/s ( 16 subframes/frame)
"""


class d06(Config):
    def __init__(self) -> None:
        super().__init__()
        ## Sim&RawDataset
        self.lum_info = False

    def train(self) -> None: raise NotImplementedError

    def eval(self) -> None:
        super().eval()
        self.ckpt_load_path = "ckpt/d04/140"


class d06_000(d06):
    def __init__(self) -> None:
        super().__init__()
        self.threshold = 0.000

    def eval(self) -> None:
        super().eval()
        self.data_save_fold = "data/d-artefact/06/000"


class d06_010(d06):
    def __init__(self) -> None:
        super().__init__()
        self.threshold = 0.010

    def eval(self) -> None:
        super().eval()
        self.data_save_fold = "data/d-artefact/06/010"


class d06_020(d06):
    def __init__(self) -> None:
        super().__init__()
        self.threshold = 0.020

    def eval(self) -> None:
        super().eval()
        self.data_save_fold = "data/d-artefact/06/020"


class d06_030(d06):
    def __init__(self) -> None:
        super().__init__()
        self.threshold = 0.030

    def eval(self) -> None:
        super().eval()
        self.data_save_fold = "data/d-artefact/06/030"


class d06_040(d06):
    def __init__(self) -> None:
        super().__init__()
        self.threshold = 0.040

    def eval(self) -> None:
        super().eval()
        self.data_save_fold = "data/d-artefact/06/040"


class d06_050(d06):
    def __init__(self) -> None:
        super().__init__()
        self.threshold = 0.050

    def eval(self) -> None:
        super().eval()
        self.data_save_fold = "data/d-artefact/06/050"


class d06_060(d06):
    def __init__(self) -> None:
        super().__init__()
        self.threshold = 0.060

    def eval(self) -> None:
        super().eval()
        self.data_save_fold = "data/d-artefact/06/060"


class d06_070(d06):
    def __init__(self) -> None:
        super().__init__()
        self.threshold = 0.070

    def eval(self) -> None:
        super().eval()
        self.data_save_fold = "data/d-artefact/06/070"


class d06_080(d06):
    def __init__(self) -> None:
        super().__init__()
        self.threshold = 0.080

    def eval(self) -> None:
        super().eval()
        self.data_save_fold = "data/d-artefact/06/080"


class d06_090(d06):
    def __init__(self) -> None:
        super().__init__()
        self.threshold = 0.090

    def eval(self) -> None:
        super().eval()
        self.data_save_fold = "data/d-artefact/06/090"


class d06_100(d06):
    def __init__(self) -> None:
        super().__init__()
        self.threshold = 0.100

    def eval(self) -> None:
        super().eval()
        self.data_save_fold = "data/d-artefact/06/100"


""" continue train with raw data
For checkbox problem, in 5, we continue to train the 3D UNet with the raw data. 
Since there are huge number of checkbox, the training process will forcus on the 
checkbox problem first. 

Result:
After we train to checkpoint 150, the result show that the network become super 
unstable and the prediction is similar to the result of checkpoint 1, which 
means that the network the network train from the start point. Also, 
this may not be a good solution since we have to train the network every time 
we have new raw data.

up sampling rate: [4, 4, 4]
features number : [1, 16, 32]
Trainable paras : 70,353
Training   speed: 1.22 steps /s ( 10 iterations/step)
Validation speed: 2.00 steps /s ( 10 iterations/step)
Evaluation speed: 3.80 frames/s ( 16 subframes/frame)
"""


class d05(Config):
    def __init__(self) -> None:
        super().__init__()
        ## Sim&RawDataset
        self.lum_info = False

    def train(self) -> None:
        super().train()
        ## getDataLoader
        self.type_data = ["Raw", "Raw"]

        ## Train
        self.ckpt_save_fold = "ckpt/d05"
        self.ckpt_load_path = "ckpt/d04/140"
        self.ckpt_load_lr   = True

    def eval(self) -> None:
        super().eval()
        ## Eval
        self.ckpt_load_path = "ckpt/d05/150"
        self.data_save_fold = "data/d-artefact/05"


""" reduce features number
Before move to [4, 8, 8] upsampling rate, we want to try the 3D UNet with less
number of features, i.e., 25% of 1&2 used, on [4, 4, 4] so that the [4, 8, 8]
network can fit in the GPU memory. 

Since 1&2 get the prediction successfully, we may expect to use same training
step in 3&4 except the new number of features. However, from the running log of
3, the learning rate is too large according to previous experiment. Thus, in 4,
we reduce the learning rate to 5e-5. In fact, we did not use same training steps
in 4 as 1&2, i.e., split the training into two part with different learning
rate. 

Result:
The result of 4 is compareable to 1&2 except the checkboard artifacts
become more serious.

up sampling rate: [4, 4, 4]
features number : [1, 16, 32]
Trainable paras : 70,353
Training   speed: 1.22 steps /s ( 10 iterations/step)
Validation speed: 2.00 steps /s ( 10 iterations/step)
Evaluation speed: 3.80 frames/s ( 16 subframes/frame)
"""


class d04(Config):
    def __init__(self) -> None:
        super().__init__()
        ## Sim&RawDataset
        self.lum_info = False

    def train(self) -> None:
        super().train()
        ## Train
        self.lr = 5e-5
        self.ckpt_save_fold = "ckpt/d04"

    def eval(self) -> None:
        super().eval()
        ## Eval
        self.ckpt_load_path = "ckpt/d04/140"
        self.data_save_fold = "data/d-artefact/04"


class d03(Config):
    def __init__(self) -> None:
        super().__init__()
        ## Sim&RawDataset
        self.lum_info = False

    def train(self) -> None:
        super().train()
        ## Train
        self.ckpt_save_fold = "ckpt/d03"

    def eval(self) -> None: raise NotImplementedError


"""
For 1&2, we use the 3D UNet without residual and CBAM and the upsample rate
is [4, 4, 4], i.e., the pixel size is 32.5 nm in XYZ. 

For 1, we train the model using learning rate 1e-4 but the result goes to all
dark where the turning point is after 8 epoch. So we load the ckpt 8 and retrain
the 3D UNet with lr at 1e-5.

Result:
The result all match the ground truth except there are checkboard artifacts.
They appear in every direction and happean in period of 9 pixels, i.e., 1 pixel
light, 7 pixels dark, 1 pixel light. After check the raw data we use to predict,
these checkbox already exist in the raw data (low resolution) where the period
is 1. These checkbox is very dark but follow the Gaussian distribution, i.e., 
a very small Gaussian point with std around 1.7. So we can not solve this
problem by simply limit the std or lum of our simulation in some range. 

Note:
The result of 02 has been deleted since 04 and further test get similar result.

up sampling rate: [4, 4, 4]
features number : [1, 32, 64]
Trainable paras : 279,969
Training   speed: 0.75 steps /s (10 iterations/step)
Validation speed:      steps /s
Evaluation speed:      frames/s
"""


class d02(Config):
    def __init__(self) -> None:
        super().__init__()
        ## ResAttUNet
        self.feats = [1, 32, 64]

        ## Sim&RawDataset
        self.lum_info = False

    def train(self) -> None:
        super().train()
        ## Train
        self.lr = 1e-5
        self.ckpt_save_fold = "ckpt/d02"
        self.ckpt_load_path = "ckpt/d01/8"

    def eval(self) -> None:
        super().eval()
        ## getDataLoader
        self.batch_size  = 4
        self.num_workers = 2
        
        ## Eval
        self.ckpt_load_path = "ckpt/d02/140"
        self.data_save_fold = "data/d-artefact/02"


class d01(Config):
    def __init__(self) -> None:
        super().__init__()
        ## ResAttUNet
        self.feats = [1, 32, 64]

        ## Sim&RawDataset
        self.lum_info = False

    def train(self) -> None:
        super().train()
        ## Train
        self.ckpt_save_fold = "ckpt/d01"

    def eval(self) -> None:
        super().eval()
        ## getDataLoader
        self.num = [1000 * 16]
        self.batch_size  = 4
        self.num_workers = 2

        ## Eval
        self.ckpt_load_path = "ckpt/d01/8"
        self.data_save_fold = "data/d-artefact/01"
