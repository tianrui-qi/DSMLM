class Config:
    def __init__(self):
        # ========================== config for train ==========================

        self.max_epoch  = 100
        self.batch_size = 4         # for dataloader
        self.lr         = 0.001     # initial learning rate (lr)
        self.gamma      = 0.95      # for exponential lr scheduler
        self.patience   = 15        # for early stopping
        self.load       = False     # for load checkpoint or not
        self.checkpoint_path = "checkpoints"

        # ======================== config for criterion ========================

        self.kernel_size  = 7   # kernel size of GaussianBlur
        self.kernel_sigma = 1   # sigma of kernel

        # ========================= config for dataset =========================

        # number of data
        self.num_train  = 2000          # number of training data
        self.num_valid  = 500          # number of validation data
        # dimensional config that need to consider memory
        self.dim_frame  = [32, 32, 32]  # [C, H, W], by pixel
        self.up_sample  = [4, 8, 8]     # [C, H, W], by scale
        # config for adjust distribution of molecular
        self.mol_range  = [0, 12]       # min, max number of molecular per frame
        self.std_range  = [0.5, 3]      # adjust variance of molecular, by pixel
        self.lum_range  = [1/32, 1]     # since input of net is normalized
        # config for reducing resolution and adding noise
        self.bitdepth   = 12
        self.qe         = 0.82
        self.sen        = 5.88
        self.noise_mu   = 16    # mu of gaussian noise, by 2^bitdepth
        self.noise_var  = 16    # variance of gaussian noise, by 2^bitdepth
