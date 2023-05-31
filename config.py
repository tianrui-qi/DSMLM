class Paras:
    def __init__(self):
        # Parameters for netLoader
        self.CheckpointDir = "checkpoints"
        self.Checkpoint    = False
        self.WhichNet      = "unet"   # "cnn" / "unet" / "cnnFocal"
        
        # Parameters for dataLoader and dataGenerator
        self.DataDir       = "generated"
        self.NumSample     = 6000
        self.NumTrain      = 5600
        self.Noised        = True  # Noised ? noised sample : clean sample
        self.Binary        = False  # Binary ? classification : regression

        # Parameters for dataGeneratorHelper
        # dimensional parameters that need to consider memory
        self.NumMolecule   = 32  # big affect on running time
        self.NumFrame      = 20  # generate NumFrame each time
        self.DimFrame      = [64, 64, 64]  # row-column-(depth); yx(z)
        self.UpSampling    = [8,  8,  4]
        # parameters that adjust distribution of sample
        self.PixelSize     = [65, 65, 100]  # use to correct the covariance
        self.StdRange      = [0.5,   2]  # adjust covariance of moleculars
        self.LumRange      = [1/512, 1]
        self.AppearRange   = [0/16,  5/16]  # min, max % of moleculars/frame
        # parameters for adding noise
        self.noise_mu      = 0
        self.noise_var     = 1/512
