import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import Tensor

import os
import tifffile
import scipy.io

from typing import Tuple


class SimDataset(Dataset):
    def __init__(self, config, num: int) -> None:
        super(SimDataset, self).__init__()
        self.num = num

        # dimensional config
        self.dim_frame = Tensor(config.dim_frame).int()     # [D]
        self.up_sample = Tensor(config.up_sample).int()     # [D]
        self.dim_label = self.dim_frame * self.up_sample    # [D]

        # config for adjust distribution of molecular
        self.mol_range = Tensor(config.mol_range).int()     # [2]
        self.std_range = Tensor(config.std_range)           # [2, D]
        self.lum_range = Tensor(config.lum_range)           # [2]
        # config for reducing resolution and adding noise
        self.bitdepth    = config.bitdepth
        self.qe          = config.qe
        self.sensitivity = config.sensitivity
        self.dark_noise  = config.dark_noise

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        This function will return a pair of frame and label where frame with a
        shape of [*self.dim_frame] and label with a shape of [*self.dim_label],
        i.e., frame is in low resolution, large pixel size, and label is in high
        resolution, small pixel size. Both frame and label already normalized
        to [0, 1] and in torch.float32. Since we generate data in real time, we
        do not need to use index to get data from disk.

        Args:
            index (int): The index of the data. This argument is not used.
        
        Returns:
            frame (Tensor): A Tensor with a shape of [*self.dim_frame].
            label (Tensor): A Tensor with a shape of [*self.dim_label].
        """
        mean_set, var_set, lum_set = self.generateParas()
        frame = self.generateFrame(mean_set, var_set, lum_set)
        frame = self.generateNoise(frame)
        label = self.generateLabel(mean_set)
        return frame, label

    def __len__(self) -> int:
        """
        Return: 
            self.num (int): Total number of data.
        """
        return self.num

    def generateParas(self) -> Tuple[Tensor, Tensor, Tensor]:
        D = len(self.dim_frame)
        N = torch.randint(
            self.mol_range[0], self.mol_range[1] + 1, (1,))  # type: ignore

        # mean set, [N, D]
        mean_set = torch.rand(N, D) * (self.dim_frame - 1)
        # variance set, [N, D]
        var_set  = torch.rand(N, D)
        var_set *= self.std_range[1] - self.std_range[0]
        var_set += self.std_range[0]
        var_set  = var_set ** 2
        # luminance set, [N]
        lum_set  = torch.rand(N) * (self.lum_range[1] - self.lum_range[0])
        lum_set += self.lum_range[0]

        return mean_set, var_set, lum_set

    def generateFrame(self, mean_set, var_set, lum_set) -> Tensor:
        D = len(self.dim_frame)  # number of dimension, i.e., 2D/3D frame
        N = len(mean_set)        # number of molecular in this frame

        frame = torch.zeros(self.dim_frame.tolist())
        for m in range(N):
            # parameters of m molecular
            mean = mean_set[m]  # [D], float
            var  = var_set[m]   # [D], float
            lum  = lum_set[m]   # float

            # take a slice around the mean where the radia is 4 * std
            ra = torch.ceil(4 * torch.sqrt(var)).int()  # radius, [D]
            lo = torch.maximum(torch.round(mean) - ra, torch.zeros(D)).int()
            up = torch.minimum(torch.round(mean) + ra, self.dim_frame - 1).int()

            # build coordinate system of the slice
            index = [torch.arange(l, u+1) for l, u in zip(lo, up)] # type: ignore
            grid  = torch.meshgrid(*index, indexing='ij')
            coord = torch.stack([c.ravel() for c in grid], dim=1)

            # compute the probability density for each point/pixel in slice
            distribution = MultivariateNormal(mean, torch.diag(var))
            pdf  = torch.exp(distribution.log_prob(coord))
            pdf /= torch.max(pdf)  # normalized

            # set the luminate
            pdf *= lum

            # put the slice back to whole frame
            frame[tuple(coord.int().T)] += pdf

        return torch.clip(frame, 0, 1)  # prevent lum exceeding 1 or below 0

    def generateNoise(self, frame: Tensor) -> Tensor:
        """
        This function will add camer noise to the input frame. 
        
        We add three type of noise in our camera noise. First we convert the
        gray frame to photons to add shot noise. Then convert to electons to 
        add dark noise. Finally, we round the frame which simulate the loss of 
        information due to limit bitdepth when store data.

        Args:
            frame (Tensor): [*self.dim_frame] tensor representing the frame
                we will add noise.

        Return:
            noise (Tensor): [*self.dim_frame] tensor representing the frame
                after we add noise.
        """

        frame *= 2**self.bitdepth - 1   # clean    -> gray
        frame /= self.sensitivity       # gray     -> electons
        frame /= self.qe                # electons -> photons
        # shot noise / poisson noise
        frame  = torch.poisson(frame)
        frame *= self.qe                # photons  -> electons
        # dark noise / gaussian noise
        frame += torch.normal(0.0, self.dark_noise, size=frame.shape)
        frame *= self.sensitivity       # electons -> gray
        # reducing resolution casue by limit bitdepth when store data
        frame  = torch.round(frame)
        frame /= 2**self.bitdepth - 1  # gray     -> noised

        return torch.clip(frame, 0, 1)  # prevent lum exceeding 1 or below 0

    def generateLabel(self, mean_set: Tensor) -> Tensor:
        """
        This function will generate the label that convert the [N, D] mean_set
        representing the molecular list to [*self.dim_label] tensor where the
        location of molecular center in mean_set is 1 in label.

        Note that the mean_set and label are in different resolution, i.e., they
        have different pixel size. So we need to convert the mean_set to label
        resolution first.

        Args:
            mean_set (Tensor): [N, D] tensor representing the molecular list
                in low resolution, i.e., large pixel size.
        
        Return:
            label (Tensor): [*self.dim_label] tensor representing the label in
                high resolution, i.e., small pixel size. 
        """
        # low resolution to super resolution, i.e., decrease pixel size
        mean_set = (mean_set + 0.5) * self.up_sample - 0.5
        
        label = torch.zeros(self.dim_label.tolist())
        label[tuple(torch.round(mean_set).T.int())] = 1
        
        return torch.clip(label, 0, 1)  # prevent lum exceeding 1 or below 0


class RawDataset(Dataset):
    def __init__(self, config, num: int) -> None:
        super(RawDataset, self).__init__()
        self.num = num

        # dimensional config
        self.dim_frame = Tensor(config.dim_frame).int()     # [D]
        self.up_sample = Tensor(config.up_sample).int()     # [D]
        self.dim_label = self.dim_frame * self.up_sample    # [D]
        
        # subframe index
        self.h_range = config.h_range
        self.w_range = config.w_range
        self.num_sub_h = self.h_range[1] - self.h_range[0] + 1
        self.num_sub_w = self.w_range[1] - self.w_range[0] + 1
        self.num_sub = self.num_sub_h * self.num_sub_w
        # data path
        self.frames_load_folder = config.frames_load_folder
        self.mlists_load_folder = config.mlists_load_folder
        
        # file name list
        self.frames_list = os.listdir(self.frames_load_folder)
        self.mlists_list = os.listdir(self.mlists_load_folder)
        
        # store the current frame and mlist in memory
        self.frame = None
        self.mlist = None
        self.current_frame_index = -1

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        frame_index = index // self.num_sub        # frame index
        sub_index   = index %  self.num_sub        # subframe index
        h = sub_index // self.num_sub_w + self.h_range[0]   # height index 
        w = sub_index %  self.num_sub_w + self.w_range[0]   # width index 

        # load new frame and mlist if frame index is different
        if frame_index != self.current_frame_index: 
            self.readNext(frame_index)
            self.current_frame_index = frame_index

        # frame
        subframe = self.frame[  # type: ignore
            0      : 64, 
            h * 52 : 64 + h * 52, 
            w * 52 : 64 + w * 52,
        ]

        # mlist
        submlist = self.mlist  # type: ignore
        submlist = submlist[submlist[:, 1] >= h * 52]  # type: ignore
        submlist = submlist[submlist[:, 1] <= h * 52 + 63]
        submlist = submlist[submlist[:, 2] >= w * 52]
        submlist = submlist[submlist[:, 2] <= w * 52 + 63]
        submlist = submlist - Tensor([0, h*52, w*52])
        submlist = (submlist + 0.5) * self.up_sample - 0.5
        submlist = torch.round(submlist).int()

        # label
        sublabel = torch.zeros(*self.dim_label.tolist())
        sublabel[tuple(submlist.t())] = 1

        return subframe, sublabel

    def __len__(self) -> int:
        """
        Return: 
            self.num (int): Total number of data.
        """
        return self.num

    def readNext(self, index) -> None:
        # frame
        self.frame = torch.from_numpy(tifffile.imread(
            os.path.join(self.frames_load_folder, self.frames_list[index])
        ))
        self.frame = (self.frame / torch.max(self.frame)).float()
        self.frame = F.pad(self.frame, (10, 10, 10, 10))
        
        # mlist
        _, self.mlist = scipy.io.loadmat( # type: ignore
            os.path.join(self.mlists_load_folder, self.mlists_list[index])
        ).popitem()
        self.mlist = torch.from_numpy(self.mlist).float()
        self.mlist = self.mlist[:, [2, 0, 1]] - 1  # (H W D) -> (D H W)
        self.mlist[:, 1:] += 10  # match the coordinate of frame after padding

    def combineFrame(self, subframes: Tensor) -> Tensor:
        shape = self.up_sample * \
            Tensor([64, 52 * self.num_sub_h, 52 * self.num_sub_w]).int()
        frame = torch.zeros(
            shape.tolist(), dtype=subframes.dtype, device=subframes.device)
        
        for sub_index in range(self.num_sub):
            h = sub_index // self.num_sub_w
            w = sub_index %  self.num_sub_w

            frame[
                :,
                h * 52 * self.up_sample[1] : (h+1) * 52 * self.up_sample[1], 
                w * 52 * self.up_sample[2] : (w+1) * 52 * self.up_sample[2],
            ] = subframes[
                sub_index, :,
                6 * self.up_sample[1] : -6 * self.up_sample[1],
                6 * self.up_sample[2] : -6 * self.up_sample[2],
            ]

        return frame[
            :,
            4 * self.up_sample[1] : -4 * self.up_sample[1],
            4 * self.up_sample[2] : -4 * self.up_sample[2],
        ]


def getDataLoader(config) -> Tuple[DataLoader, ...]:
    """
    This function will return a tuple of dataloader for each dataset. Four paras
    in config will be used to create the dataloader, i.e., config.num, 
    config.type, config.batch_size, and config.num_workers. All dataloader will
    have the same batch_size and num_workers.
    
    For example:
        if the input config has
            config.num = [100, 200]
            config.type = ["Sim", "Raw"]
        the dataloader tuple return by this function will be
            dataloader = (
                SimDataset with 100 data, 
                RawDataset with 200 data
            )

    Args:
        config (Config): The config class for this project.
            config.num (List[int]): A list of int, where each int is the number
                of data in each dataset.
            config.type (List[str]): A list of str, where each str is the type
                of each dataset, i.e., "Sim" or "Raw".
            config.batch_size (int): The batch size for each dataloader.
            config.num_workers (int): The number of workers for each dataloader.
    
    Returns:
        dataloader (Tuple[DataLoader]): A tuple of dataloader for each dataset.
            Will have same length as config.num and config.type.
    """

    if len(config.num) != len(config.type): raise ValueError(
        "The length of config.num and config.type should be the same."
    )

    dataloader = []
    for d in range(len(config.num)):
        if config.type[d] == "Sim":
            dataset = SimDataset(config, config.num[d])
        elif config.type[d] == "Raw":
            dataset = RawDataset(config, config.num[d])
        else:
            raise ValueError("Only Raw and Sim dataset is supported.")
        dataloader.append(DataLoader(
            dataset,
            batch_size=config.batch_size, 
            num_workers=config.num_workers, 
            pin_memory=True
        ))
    return tuple(dataloader)


if __name__ == "__main__":
    """
    Test code for two dataset. 
    No need to run this file independently when train and evaluate the network.
    """
    from tifffile import imsave
    from config import Config

    # create dir to store test frame
    if not os.path.exists("data"): os.makedirs("data")

    # test using default config
    config = Config()

    # test the RawDataset
    dataset = RawDataset(config, 1)
    frame, label = dataset[0]
    imsave('data/RawFrame.tif', (frame * 255).to(torch.uint8).numpy())
    imsave('data/RawLabel.tif', (label * 255).to(torch.uint8).numpy())

    # test the SimDataset
    dataset = SimDataset(config, 1)
    frame, label = dataset[0]
    imsave('data/SimFrame.tif', (frame * 255).to(torch.uint8).numpy())
    imsave('data/SimLabel.tif', (label * 255).to(torch.uint8).numpy())
