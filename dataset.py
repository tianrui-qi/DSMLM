import numpy as np
from scipy.stats import multivariate_normal

import torch
from torch.utils.data import Dataset

from config import Config

class MyDataset(Dataset):
    def __init__(self, config):
        super(MyDataset, self).__init__()

        self.config = config
        

    def __getitem__(self):
        return

    def __len__(self):
        return
    

    # Help function for __getitem__

    def generateDataParas(self):
        # load parameters we will use
        dim_frame: np.ndarray = np.array(self.config.dim_frame)  # [2]
        mol_range: np.ndarray = np.array(self.config.mol_range)  # [2]
        max_std  : np.ndarray = np.array(self.config.max_std)    # [D]
        lum_range: np.ndarray = np.array(self.config.lum_range)  # [2]

        D: int = len(dim_frame)
        N: int = np.random.randint(mol_range[0], mol_range[1] + 1)

        # mu set, [D * N]
        mu_set  = np.random.rand(D, N) * (dim_frame.reshape(-1, 1) - 1)
        # variance set, [D * N]
        var_set = (np.random.rand(D, N) * max_std.reshape(-1, 1)) ** 2
        # luminance set, [N]
        lum_set = np.random.rand(N) * (lum_range[1] - lum_range[0])
        lum_set += lum_range[0]

        return mu_set, var_set, lum_set


if __name__ == "__main__":
    config = Config()
    dataset = MyDataset(config)
    dataset.generateDataParas()
