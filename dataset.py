import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.stats import multivariate_normal
from skimage.transform import resize


class SimDataset(Dataset):
    def __init__(self, config, num):
        super(SimDataset, self).__init__()
        self.num        = num
        # dimensional config that need to consider memory
        self.dim_frame  = np.array(config.dim_frame).astype(int)         # [D]
        self.up_sample  = np.array(config.up_sample).astype(int)         # [D]
        self.dim_label  = (self.dim_frame * self.up_sample).astype(int)  # [D]
        # config for adjust distribution of molecular
        self.mol_range  = np.array(config.mol_range).astype(int)         # [2]
        self.std_range  = np.array(config.std_range).astype(float)       # [2]
        self.lum_range  = np.array(config.lum_range).astype(float)       # [2]
        # config for adding camera noise
        self.bitdepth   = config.bitdepth
        self.qe         = config.qe
        self.sen        = config.sen
        self.noise_mu   = config.noise_mu
        self.noise_var  = config.noise_var

    def __getitem__(self, idx):
        mean_set, var_set, lum_set = self.generateParas()
        frame = self.generateFrame(mean_set, var_set, lum_set)  # [dim_label]
        frame = self.generateNoise(frame)                       # [dim_frame]
        label = self.generateLabel(mean_set, lum_set)           # [dim_label]

        return torch.from_numpy(frame), torch.from_numpy(label)

    def __len__(self):
        return self.num
    
    # help functions for __getitem__

    def generateParas(self):
        D = len(self.dim_label)  # number of dimension, i.e., 2D/3D frame
        N = np.random.randint(self.mol_range[0], self.mol_range[1] + 1)

        # mean set, [D * N]
        mean_set  = np.random.rand(D, N)
        mean_set *= self.dim_label.reshape(-1, 1) - 1
        # variance set, [D * N]
        var_set   = np.random.rand(D, N)
        var_set  *= self.std_range[1] - self.std_range[0]
        var_set  += self.std_range[0]
        var_set  *= self.up_sample.reshape(-1, 1) - 1
        var_set  *= var_set
        # luminance set, [N]
        lum_set   = np.random.rand(N) * (self.lum_range[1] - self.lum_range[0])
        lum_set  += self.lum_range[0]

        return mean_set, var_set, lum_set
    
    def generateFrame(self, mean_set, var_set, lum_set):
        D = len(self.dim_label)  # number of dimension, i.e., 2D/3D frame
        N = len(lum_set)         # number of molecular in this frame

        frame = np.zeros(self.dim_label)
        for m in range(N):
            # parameters of m molecular
            mean = mean_set[:,m]
            var  = var_set[:,m]
            lum  = lum_set[m]

            # take a slice around the mean where the radia is 5 * std
            ra = np.ceil(5 * np.sqrt(var))  # radius, [D]
            lo = np.floor(np.maximum(np.round(mean) - ra, np.zeros(D)))
            up = np.ceil(np.minimum(np.round(mean) + ra, self.dim_label - 1))

            # build coordinate system of the slice
            index     = [np.arange(l, u+1) for l, u in zip(lo, up)]
            grid_cell = np.meshgrid(*index, indexing='ij')  
            coord     = np.column_stack([c.ravel() for c in grid_cell])

            # compute the probability density for each point/pixel in slice
            pdf = multivariate_normal.pdf(
                coord, mean=mean, cov=np.diag(var)) # type: ignore

            # set the luminate
            pdf = pdf * lum / np.amax(pdf)

            # put the slice back to whole frame
            frame[tuple(coord.astype(int).T)] += pdf

        return np.clip(frame, 0, 1)  # prevent lum exceeding 1 or below 0
    
    def generateNoise(self, frame):
        # decrease pixel number / increase pixel size
        frame  = resize(frame, self.dim_frame)

        frame *= 2**self.bitdepth - 1  # clean    -> gray
        frame /= self.sen              # gray     -> electons
        frame /= self.qe               # electons -> photons
        # shot noise / poisson noise
        frame  = np.random.poisson(frame, size=frame.shape).astype(np.float64)
        frame *= self.qe               # photons  -> electons
        frame *= self.sen              # electons -> gray
        # dark noise / gaussian noise
        frame += np.random.normal(
            self.noise_mu  * np.random.rand(1), 
            self.noise_var * np.random.rand(1), frame.shape)
        # reducing resolution casue by limit bitdepth when store data
        frame  = np.round(frame)
        frame /= 2**self.bitdepth - 1  # gray     -> noised

        return np.clip(frame, 0, 1)  # prevent lum exceeding 1 or below 0

    def generateLabel(self, mean_set, lum_set):
        label = np.zeros(self.dim_label)
        label[tuple(np.round(mean_set).astype(int))] = lum_set
        
        return label


# test code using default config

if __name__ == "__main__":
    import os
    from tifffile import imsave
    from config import Config
    
    config = Config()
    dataset = SimDataset(config, 1)

    # print parameters of each molecular
    mean_set, var_set, lum_set = dataset.generateParas()
    np.set_printoptions(precision=2)
    for m in range(len(mean_set.T)):
        print("mol {}\tmean: {}\tvar: {}\tlum: {}".format(
            m, mean_set[:, m], var_set[:, m], lum_set[m]))
    np.set_printoptions()
    
    if not os.path.exists("tests"):os.makedirs("tests")

    frame = dataset.generateFrame(mean_set, var_set, lum_set)  # [dim_label]
    imsave('tests/dataset-frame.tif', np.array(frame*255, dtype=np.uint8))

    noise = dataset.generateNoise(frame)                       # [dim_frame]
    imsave('tests/dataset-noise.tif', np.array(noise*255, dtype=np.uint8))

    label = dataset.generateLabel(mean_set, lum_set)           # [dim_label]
    imsave('tests/dataset-label.tif', np.array(label*255, dtype=np.uint8))
