from typing import List


class Config:
    def __init__(self) -> None:
        ## ResAttUNet
        self.dim  : int = 3
        self.feats: List[int] = [1, 16, 32]
        self.use_res : bool = False
        self.use_cbam: bool = False

        ## GaussianBlurLoss
        self.kernel_size : int   = 7
        self.kernel_sigma: float = 1.0

        ## SimDataset & RawDataset
        # dimension
        self.dim_dst: List[int] = [160, 160, 160]  # [C, H, W], pixel

        ## SimDataset
        # molecular profile
        self.std_src: List[List[float]] = [   # std range
            [1.0, 1.0, 1.0],  # minimum std, [C, H, W], by pixel
            [3.0, 2.5, 2.5],  # maximum std, [C, H, W], by pixel
        ]

        ## RawDataset
        # scale up factor
        self.scale: List[int] = [4, 4, 4]   # [C, H, W]
        # data path
        self.frames_load_fold: str = "D:/frames"
        self.mlists_load_fold: str = "D:/mlists"    # set to "" if not used

    def train(self) -> None:
        ## Train
        # train
        self.accumu_steps: int = 5
        self.lr   : float = 1e-4    # initial learning rate (lr)
        self.gamma: float = 0.95    # decay rate of lr
        # checkpoint
        self.ckpt_save_fold: str  = "ckpt/default"
        self.ckpt_load_path: str  = ""        # path without .ckpt
        self.ckpt_load_lr  : bool = False     # load lr from ckpt
        # dataloader
        self.num: List[int] = [10000, 5000]   # train and valid
        self.batch_size : int = 2

    def eval(self) -> None:
        ## Eval - also use some config of train
        # checkpoint
        self.ckpt_load_path: str = ""   # path without .ckpt
        # data
        self.data_save_fold: str = "data/default"
        # dataloader
        self.batch_size : int = 8
