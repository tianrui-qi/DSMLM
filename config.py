class Config:
    def __init__(self):
        self.num_sample = 10000
        self.num_train  = 7000
        # dimensional config that need to consider memory
        self.dim_frame  = [64, 64, 64]  # row-column-(depth)
        self.up_sample  = [8, 8, 4]
        # config for adjust distribution of molecular
        self.mol_range  = [0, 16]  # min, max number of molecular per frame
        self.std_range  = [0.5, 3]  # adjust variance of molecular
        self.lum_range  = [1/512, 1]
        # config for reducing resolution and adding noise
        self.bitdepth   = 16
        self.qe         = 0.82
        self.sen        = 5.88
        self.noise_mu   = self.lum_range[0]
        self.noise_var  = 1/128
