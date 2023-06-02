import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import zoom
from scipy.stats import multivariate_normal
from typing import Tuple  # for type annotations only

class SimDataset(Dataset):
    def __init__(self, config, num) -> None:
        super(SimDataset, self).__init__()
        self.num        = num
        # dimensional config that need to consider memory
        self.dim_frame  = np.array(config.dim_frame)    # [D]
        self.up_sample  = np.array(config.up_sample)    # [D]
        # config for adjust distribution of molecular
        self.mol_range  = np.array(config.mol_range)    # [2]
        self.std_range  = np.array(config.std_range)    # [2]
        self.lum_range  = np.array(config.lum_range)    # [2]
        # config for adding camera noise
        self.bitdepth   = config.bitdepth
        self.qe         = config.qe
        self.sen        = config.sen
        self.noise_mu   = config.noise_mu
        self.noise_var  = config.noise_var

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_set, var_set, lum_set = self.generateParas()
        frame = self.generateFrame(mean_set, var_set, lum_set)  # [dim_frame]
        frame = self.generateNoise(frame)                       # [dim_frame]
        frame = self.generateSclup(frame)                       # [dim_frame_up]
        label = self.generateLabel(mean_set, lum_set)           # [dim_frame_up]

        return torch.from_numpy(frame), torch.from_numpy(label)

    def __len__(self) -> int:
        return self.num
    
    # Help function for __getitem__

    def generateParas(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        D = len(self.dim_frame)  # number of dimension, i.e., 2D/3D frame
        N = np.random.randint(self.mol_range[0], self.mol_range[1] + 1)

        # mean set, [D * N]
        mean_set = np.random.rand(D, N) * (self.dim_frame.reshape(-1, 1) - 1)
        # variance set, [D * N]
        var_set  = np.random.rand(D, N)
        var_set *= self.std_range[1] - self.std_range[0]
        var_set += self.std_range[0]
        var_set *= var_set
        # luminance set, [N]
        lum_set  = np.random.rand(N) * (self.lum_range[1] - self.lum_range[0])
        lum_set += self.lum_range[0]

        return mean_set, var_set, lum_set

    def generateFrame(self, mean_set, var_set, lum_set) -> np.ndarray:
        D = len(self.dim_frame)  # number of dimension, i.e., 2D/3D frame
        N = len(lum_set)         # number of molecular in this frame

        frame = np.zeros(self.dim_frame)
        for m in range(N):
            # parameters of m molecular
            mean = mean_set[:,m]
            var  = var_set[:,m]
            lum  = lum_set[m]

            # take a slice around the mean where the radia is 5 * std
            ra = np.ceil(5 * np.sqrt(var))  # radius, [D]
            lo = np.floor(np.maximum(np.round(mean) - ra, np.zeros(D)))
            up = np.ceil(np.minimum(np.round(mean) + ra, self.dim_frame - 1))

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
    
    def generateNoise(self, frame) -> np.ndarray:
        frame *= 2**self.bitdepth - 1  # clean    -> gray
        frame /= self.sen              # gray     -> electons
        frame /= self.qe               # electons -> photons
        # shot noise / poisson noise
        frame = np.random.poisson(frame, size=frame.shape).astype(np.float64)
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

    def generateSclup(self, frame) -> np.ndarray:
        frame = zoom(frame, self.up_sample, order=2)

        return np.clip(frame, 0, 1)  # prevent lum exceeding 1 or below 0

    def generateLabel(self, mean_set, lum_set) -> np.ndarray:
        dim_frame_up = self.dim_frame * self.up_sample
        mean_set_up  = mean_set * self.up_sample.reshape(-1, 1)
        
        label = np.zeros(dim_frame_up.astype(int))
        label[tuple(np.round(mean_set_up).astype(int))] = lum_set
        
        return label
