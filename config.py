class Config:
    def __init__(self):
        # ========================== config for train ==========================

        # train
        self.max_epoch  = 400
        # learning rate
        self.lr    = 0.0001     # initial learning rate (lr)
        self.gamma = 0.95
        # checkpoint
        self.checkpoint_path = "checkpoints"  # checkpoints path without .pt
        self.save_pt_epoch = False  # save pt every epoch with epoch idx

        # ======================== config for criterion ========================

        self.kernel_size  = 7       # kernel size of GaussianBlur
        self.kernel_sigma = 1.0     # sigma of kernel
        self.l1_coeff     = 0.0     # to repeat deep storm loss function, set 1

        # ========================== config for data ===========================

        # number of data
        self.batch_size  = 1    # for dataloader
        self.num_workers = 1    # for dataloader
        self.num_train = 8000   # number of training data
        self.num_valid = 2000   # number of validation data
        # dimensional config that need to consider memory
        self.dim_frame = [32, 32, 32]   # [C, H, W], by pixel
        self.up_sample = [4, 4, 4]      # [C, H, W], by scale
        self.mol_epoch = 128            # num of molecular simulated per epoch
        # config for adjust distribution of molecular
        self.mol_range = [0, 64]        # min, max number of molecular per frame
        self.std_range = [0.5, 3.0]     # adjust variance of molecular, by pixel
        self.lum_range = [1/32, 1.0]    # since input of net is normalized
        # config for reducing resolution and adding noise
        self.bitdepth  = 12
        self.qe        = 0.82
        self.sen       = 5.88
        self.noise_mu  = 0.0    # mu of gaussian noise, by 2^bitdepth
        self.noise_var = 0.0    # variance of gaussian noise, by 2^bitdepth
