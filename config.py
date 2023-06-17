class Config:
    def __init__(self):
        # ========================== config for train ==========================

        # train
        self.max_epoch  = 400
        self.batch_size = 1        # for dataloader
        # learning rate
        self.lr         = 0.001    # initial learning rate (lr)
        self.factor     = 0.5      # for scheduler
        self.patience   = 5        # for scheduler
        # running log
        self.logdir     = None
        # checkpoint
        self.load       = False     # for load checkpoint or not
        self.checkpoint_path = "checkpoints"  # checkpoints path without .pt
        self.save_pt_epoch = False  # save pt every epoch with epoch idx

        # ======================== config for criterion ========================

        self.kernel_size  = 7     # kernel size of GaussianBlur
        self.kernel_sigma = 1.0   # sigma of kernel
        self.l1_coeff     = 0.0   # to repeat deep storm loss function, set 1

        # ========================= config for dataset =========================

        # number of data
        self.num_train  = 7000          # number of training data
        self.num_valid  = 3000          # number of validation data
        # dimensional config that need to consider memory
        self.dim_frame  = [64, 64, 64]  # [C, H, W], by pixel
        self.up_sample  = [4, 8, 8]     # [C, H, W], by scale
        # config for adjust distribution of molecular
        self.mol_range  = [0, 12]       # min, max number of molecular per frame
        self.std_range  = [0.5, 3.0]    # adjust variance of molecular, by pixel
        self.lum_range  = [1/32, 1.0]   # since input of net is normalized
        # config for reducing resolution and adding noise
        self.bitdepth   = 12
        self.qe         = 0.82
        self.sen        = 5.88
        self.noise_mu   = 0    # mu of gaussian noise, by 2^bitdepth
        self.noise_var  = 0    # variance of gaussian noise, by 2^bitdepth
