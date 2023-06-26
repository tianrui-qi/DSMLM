from typing import List


class Config:
    def __init__(self):
        # dimensional config
        # MUST be same accross whole pipline
        self.dim_frame: List[int] = [64, 64, 64]    # [C, H, W], by pixel
        self.up_sample: List[int] = [ 4,  4,  4]    # [C, H, W], by scale

        # ========================= config for train ========================= #

        # train
        self.max_epoch: int = 400
        # learning rate
        self.lr   : float = 0.0001     # initial learning rate (lr)
        self.gamma: float = 0.95
        # checkpoint
        self.checkpoint_path = "checkpoints"  # checkpoints path without .pt
        self.save_pt_epoch   = False    # save pt every epoch with epoch idx

        # ========================= config for model== ======================= #

        self.kernel_size : int = 7      # kernel size of GaussianBlur
        self.kernel_sigma: float = 1.0  # sigma of kernel
        self.l1_coeff    : float = 0.0  # set 1 to repeat deep storm

        # ========================= config for data ========================== #

        # number of data
        self.num        : List[int] = [8000, 2000]  # num of train, valid data
        self.batch_size : int       = 1             # for dataloader
        self.num_workers: int       = 1             # for dataloader

        ## For SimDataset
        # config for adjust distribution of molecular
        self.mol_epoch: int         = 128   # num molecular simulated per epoch
        self.mol_range: List[int]   = [0, 16]       # min, max num of mol/frame
        self.std_range: List[float] = [0.5, 3.0]    # by pixel in low resolution
        self.lum_range: List[float] = [1/32, 1.0]
        # config for reducing resolution and adding noise
        self.bitdepth : int   = 8
        self.qe       : float = 0.82
        self.sen      : float = 5.88
        self.noise_mu : float = 0.0    # mu of Gaussian noise, by 2^bitdepth
        self.noise_var: float = 0.0    # variance of dark noise, by 2^bitdepth

        ## For RawDataset
        self.raw_folder  = "data/raw"
