class Config:
    def __init__(self):
        self.num_train  = 700
        self.num_valid  = 300
        self.max_epoch  = 100
        self.batch_size = 2
        self.lr         = 0.0001  # initial lr
        self.gamma      = 0.95   # for exponential lr scheduler
        self.patience   = 10
        self.checkpoint_path = "checkpoints.pt"

        # dimensional config that need to consider memory
        self.dim_frame  = [64, 48, 48]  # [C, H, W]
        self.up_sample  = [4, 8, 8]
        # config for adjust distribution of molecular
        self.mol_range  = [0, 16]    # min, max number of molecular per frame
        self.std_range  = [0.5, 3]   # adjust variance of molecular
        self.lum_range  = [1/32, 1]  # since input of net is normalized
        # config for reducing resolution and adding noise
        self.bitdepth   = 12
        self.qe         = 0.82
        self.sen        = 5.88
        self.noise_mu   = 32
        self.noise_var  = 32
