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

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        mean_set, var_set, lum_set = self._generateParas()

        frame = self._generateFrame(mean_set, var_set, lum_set)
        frame = self.addCameraNoise(frame)
        frame = F.interpolate(
            frame.unsqueeze(0).unsqueeze(0),
            scale_factor=self.up_sample.tolist()
        ).squeeze(0).squeeze(0)

        label = self._generateLabel(mean_set)

        return frame, label

    def __len__(self) -> int:
        """
        Return: 
            self.num (int): Total number of data.
        """
        return self.num

    def _generateParas(self) -> Tuple[Tensor, Tensor, Tensor]:
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

    def _generateFrame(self, mean_set, var_set, lum_set) -> Tensor:
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

    def _generateLabel(self, mean_set: Tensor) -> Tensor:
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

    @staticmethod
    def addCameraNoise(
        frame: Tensor,
        bitdepth: int = 16, qe: float = 0.82,
        sensitivity: float = 5.88, dark_noise: float = 2.29
    ) -> Tensor:
        frame *= 2**bitdepth - 1    # clean    -> gray
        frame /= sensitivity        # gray     -> electons
        frame /= qe                 # electons -> photons
        # shot noise / poisson noise
        frame  = torch.poisson(frame)
        frame *= qe                 # photons  -> electons
        # dark noise / gaussian noise
        frame += torch.normal(0.0, dark_noise, size=frame.shape)
        frame *= sensitivity        # electons -> gray
        # reducing resolution casue by limit bitdepth when store data
        frame  = torch.round(frame)
        frame /= 2**bitdepth - 1    # gray     -> noised

        return torch.clip(frame, 0, 1)  # prevent lum exceeding 1 or below 0


class RawDataset(Dataset):
    def __init__(self, config, num: int) -> None:
        super(RawDataset, self).__init__()
        self.num = num

        # dimensional config
        self.dim_frame = Tensor(config.dim_frame).int()     # [D]
        self.up_sample = Tensor(config.up_sample).int()     # [D]
        self.dim_label = self.dim_frame * self.up_sample    # [D]

        # read option
        self.threshold = config.threshold

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
            self._readNext(frame_index)
            self.current_frame_index = frame_index

        # frame
        subframe = self.frame[  # type: ignore
            :, 
            h * 32 : 40 + h * 32, 
            w * 32 : 40 + w * 32,
        ]
        subframe = F.interpolate(
            subframe.unsqueeze(0).unsqueeze(0), 
            scale_factor=self.up_sample.tolist()
        ).squeeze(0).squeeze(0)

        # mlist
        submlist = self.mlist  # type: ignore
        submlist = submlist[submlist[:, 1] >= h * 32]  # type: ignore
        submlist = submlist[submlist[:, 1] <= h * 32 + 39]
        submlist = submlist[submlist[:, 2] >= w * 32]
        submlist = submlist[submlist[:, 2] <= w * 32 + 39]
        submlist = submlist - Tensor([0, h*32, w*32])
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

    def _readNext(self, index: int, pad: int = 4) -> None:
        # frame
        self.frame = torch.from_numpy(tifffile.imread(
            os.path.join(self.frames_load_folder, self.frames_list[index])
        ))
        self.frame = (self.frame / torch.max(self.frame)).float()
        self.frame = F.interpolate(
            self.frame.unsqueeze(0).unsqueeze(0), 
            size = (32, 512, 512)
        ).squeeze(0).squeeze(0)
        self.frame = F.pad(self.frame, (pad, pad, pad, pad, pad, pad))
        self.frame[self.frame < self.threshold] = 0

        # mlist
        _, self.mlist = scipy.io.loadmat( # type: ignore
            os.path.join(self.mlists_load_folder, self.mlists_list[index])
        ).popitem()
        self.mlist = torch.from_numpy(self.mlist).float()
        self.mlist = self.mlist[:, [2, 0, 1]] - 1  # (H W D) -> (D H W)
        self.mlist[:, 0] = (self.mlist[:, 0] + 0.5) / 2 - 0.5
        self.mlist += pad   # match the coordinate of frame after padding

    def combineFrame(self, subframes: Tensor, pad: int = 4) -> Tensor:
        shape = self.up_sample * \
            Tensor([32, 32 * self.num_sub_h, 32 * self.num_sub_w]).int()
        frame = torch.zeros(
            shape.tolist(), dtype=subframes.dtype, device=subframes.device
        )

        for sub_index in range(self.num_sub):
            h = sub_index // self.num_sub_w
            w = sub_index %  self.num_sub_w

            frame[
                :,
                h * 32 * self.up_sample[1] : (h+1) * 32 * self.up_sample[1], 
                w * 32 * self.up_sample[2] : (w+1) * 32 * self.up_sample[2],
            ] = subframes[
                sub_index,
                pad * self.up_sample[0] : -pad * self.up_sample[0],
                pad * self.up_sample[1] : -pad * self.up_sample[1],
                pad * self.up_sample[2] : -pad * self.up_sample[2],
            ]

        return frame


def getDataLoader(config) -> Tuple[DataLoader, ...]:
    if len(config.type_data) != len(config.num): raise ValueError(
        "The length of config.num and config.type_data should be the same."
    )

    dataloader = []
    for d in range(len(config.num)):
        if config.type_data[d] == "Sim":
            dataset = SimDataset(config, config.num[d])
        elif config.type_data[d] == "Raw":
            dataset = RawDataset(config, config.num[d])
        else:
            raise ValueError(f"Unsupported data: {config.type_data[d]}")
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
    if not os.path.exists("data/test"): os.makedirs("data/test")

    # test using default config
    config = Config()
    config.train()

    # test the RawDataset
    dataset = RawDataset(config, 5000)
    frame, label = dataset[0]
    imsave('data/test/RawFrame.tif', frame.numpy())
    imsave('data/test/RawLabel.tif', label.numpy())

    # density of RawDataset
    num = 0
    for i in range(5000):
        frame, label = dataset[i]
        num += len(torch.nonzero(label))
    print("density of RawDataset:", num / 5000)

    # test the SimDataset
    dataset = SimDataset(config, 1)
    frame, label = dataset[0]
    print("Number of molecular:", len(torch.nonzero(label)))
    imsave('data/test/SimFrame.tif', frame.numpy())
    imsave('data/test/SimLabel.tif', label.numpy())
