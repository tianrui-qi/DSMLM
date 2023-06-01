class Config:
    def __init__(self):
        # Parameters for dataLoader and dataGenerator
        self.num_sample  = 6000
        self.num_train   = 5600
        self.noised     = True  # Noised ? noised sample : clean sample
        self.binary     = False  # Binary ? classification : regression

        # Parameters for dataGeneratorHelper
        # dimensional parameters that need to consider memory
        self.dim_frame   = [64, 64, 64]  # row-column-(depth); yx(z)
        self.up_sampling = [8, 8, 4]
        # parameters that adjust distribution of sample
        self.mol_range   = [0, 16]  # min, max number of moleculars per frame
        self.max_std     = [3, 3, 3]  # adjust variance of moleculars
        self.lum_range   = [1/512, 1]
        # parameters for adding noise
        self.noise_mu   = 0
        self.noise_var  = 1/512
