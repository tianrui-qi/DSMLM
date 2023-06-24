from typing import List


class Config:
    def __init__(self):
        # dimensional config
        # MUST be same accross simulated data, raw data, and model
        self.dim_frame: List[int] = [64, 64, 64]    # [C, H, W], by pixel
        self.up_sample: List[int] = [ 4,  8,  8]    # [C, H, W], by scale

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
        self.num: List[int] = [8000, 2000]  # number of train, valid data
        self.batch_size : int = 1        # for dataloader
        self.num_workers: int = 1        # for dataloader
        
        ## For Simulated Data (SimDataset)
        # config for adjust distribution of molecular
        self.mol_epoch: int = 128   # num of molecular simulated per epoch
        self.mol_range: List[int]   = [0, 64]       # min, max num of mol/frame
        self.std_range: List[float] = [0.5, 3.0]    # by pixel in low resolution
        self.lum_range: List[float] = [1/32, 1.0]
        # config for reducing resolution and adding noise
        self.bitdepth : int   = 12
        self.qe       : float = 0.82
        self.sen      : float = 5.88
        self.noise_mu : float = 16.0    # mu of gaussian noise, by 2^bitdepth
        self.noise_var: float = 16.0    # variance of dark noise, by 2^bitdepth

        ## For Raw Data (prepareRawData, RawDataset)
        self.raw_folder = "data/raw"
