import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import Tensor

import os
import tifffile
import scipy.io
from typing import List, Tuple, Union


class _SimDataset(Dataset):
    def __init__(
        self, num: int, 
        dim_nm_src: List[float], 
        dim_px_dst: List[int],
        std_nm_src: List[List[float]],
        scale_list: List[int], 
    ) -> None:
        super(_SimDataset, self).__init__()
        self.num = num
        # config for dimension
        self.D = len(dim_nm_src)    # number of dimension 
        self.dim_nm_src = Tensor(dim_nm_src)                # [D], float
        self.dim_px_src = None                              # [D], int
        self.dim_nm_dst = None                              # [D], float
        self.dim_px_dst = Tensor(dim_px_dst).int()          # [D], int
        # config for molecular profile
        self.std_nm_src = Tensor(std_nm_src)                # [2, D], float
        self.std_px_src = self.std_nm_src/self.dim_nm_src   # [2, D], float
        # config for scale up factor 
        self.scale_list = Tensor(scale_list).int()          
        self.scale      = None                              # [D], int

        # store molecular list for current frame
        self.N          = None      # number of molecula    
        self.mean_set   = None                              # [N, D], float
        self.vars_set   = None                              # [N, D], float
        self.peak_set   = None                              # [N], float

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        # generate molecular list for current frame
        self._generateMlist()

        frame = torch.zeros(self.dim_px_src.tolist())
        label = torch.zeros(self.dim_px_dst.tolist())
        for m in range(self.N):
            ## mlist
            mean = self.mean_set[m]     # [D], float
            vars = self.vars_set[m]     # [D], float
            peak = self.peak_set[m]     # float

            ## frame
            # take a slice around the mean where the radia is std*4
            ra = torch.ceil(torch.sqrt(vars)*4).int()   # radius, [D]
            lo = torch.maximum(torch.round(mean)-ra, torch.zeros(self.D)).int()
            up = torch.minimum(torch.round(mean)+ra, frame.shape-1).int()
            # build coordinate system of the slice
            index = [torch.arange(l, u+1) for l, u in zip(lo, up)]
            grid  = torch.meshgrid(*index, indexing='ij')
            coord = torch.stack([c.ravel() for c in grid], dim=1)
            # compute the probability density for each point/pixel in slice
            distribution = MultivariateNormal(mean, torch.diag(vars))
            pdf  = torch.exp(distribution.log_prob(coord))
            pdf /= torch.max(pdf)  # normalized
            # set the luminate, peak of the gaussian/molecular
            pdf *= peak
            # put the slice back to whole frame
            frame[tuple(coord.int().T)] += pdf

            ## label
            # dim_px_src -> dim_px_dst
            mean = (mean + 0.5) * self.scale - 0.5  # [D], float
            mean = torch.round(mean).int()          # [D], int
            # set the brightness to the integral of gaussian/molecular
            label[tuple(mean)] += peak * torch.sqrt(torch.prod(vars))

        # add noise
        save_bit = 2**(3+torch.randint(0, 3, (1,)))     # 8, 16, or 32
        frame = self._generateNoise(frame, save_bit=save_bit)
        # dim_px_src -> dim_px_dst
        frame = F.interpolate(
            frame.unsqueeze(0).unsqueeze(0),
            scale_factor=self.scale.tolist()
        ).squeeze(0).squeeze(0)

        return torch.clip(frame, 0, 1), label

    def __len__(self) -> int:
        """
        Return: 
            self.num (int): Total number of data.
        """
        return self.num

    def _generateMlist(self) -> None:
        # random scale up factor
        self.scale = torch.randint(0, len(self.scale_list), (self.D,))
        self.scale = self.scale_list[self.scale]                # [D], int

        # pixel size of destination frame
        self.dim_nm_dst = (self.dim_nm_src / self.scale)        # [D], float
        # pixel number of source frame
        self.dim_px_src = (self.dim_px_dst / self.scale).int()  # [D], int

        # number of molecular
        self.N = torch.sum(self.dim_px_src).int()   # max possible # of mol
        self.N = torch.randint(0, self.N+1, (1,))   # random mol # from 0 to max

        # generate parameters in source dimension
        # moleculars' mean, [N, D]
        self.mean_set  = torch.rand(self.N, self.D) * (self.dim_px_src - 1)
        # moleculars' variance, [N, D]
        self.vars_set  = torch.rand(self.N, self.D)
        self.vars_set *= self.std_px_src[1] - self.std_px_src[0]
        self.vars_set += self.std_px_src[0]
        self.vars_set  = self.vars_set ** 2
        # moleculars' peak, [N]
        self.peak_set  = torch.rand(self.N)

    def _generateNoise(
        self, frame: Tensor,
        save_bit: int = 16, camera_bit: int = 16, 
        qe: float = 0.82, sensitivity: float = 5.88, dark_noise: float = 2.29
    ) -> Tensor:
        ## camera noise
        frame *= 2**camera_bit-1    # clean    -> gray
        frame /= sensitivity        # gray     -> electons
        frame /= qe                 # electons -> photons
        # shot noise / poisson noise
        frame  = torch.poisson(frame)
        frame *= qe                 # photons  -> electons
        # dark noise / gaussian noise
        frame += torch.normal(0.0, dark_noise, size=frame.shape)
        frame *= sensitivity        # electons -> gray
        frame /= 2**camera_bit-1    # gray     -> noised

        ## noise casue by limit bitdepth when store data
        frame *= 2**save_bit-1      # clean    -> gray
        frame  = torch.round(frame)
        frame /= 2**save_bit-1      # gray     -> noised

        return frame


class _RawDataset(Dataset):
    def __init__(self, config, num: int) -> None:
        super(_RawDataset, self).__init__()
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
        self.frames_load_fold = config.frames_load_fold
        self.mlists_load_fold = config.mlists_load_fold
        # file name list
        self.frames_list = os.listdir(self.frames_load_fold)
        if self.mlists_load_fold != "":     # don't need or have mlists
            self.mlists_list = os.listdir(self.mlists_load_fold)
        # read option
        self.averagemax = self._getAveragemax()  # for normalizing all frames

        # store the current frame and mlist in memory
        self.frame = None
        self.mlist = None
        self.current_frame_index = -1

    def __getitem__(self, index: int) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        frame_index = index // self.num_sub        # frame index
        sub_index   = index %  self.num_sub        # subframe index
        h = sub_index // self.num_sub_w + self.h_range[0]   # height index 
        w = sub_index %  self.num_sub_w + self.w_range[0]   # width index 

        # load new frame and mlist if frame index is different
        if frame_index != self.current_frame_index: 
            self._readNext(frame_index)
            self.current_frame_index = frame_index

        # frame
        subframe = self.frame[
            :,
            h * 32 : 40 + h * 32,
            w * 32 : 40 + w * 32,
        ]
        subframe = F.interpolate(
            subframe.unsqueeze(0).unsqueeze(0), 
            scale_factor=self.up_sample.tolist()
        ).squeeze(0).squeeze(0)
        subframe = torch.clip(subframe, 0, 1)

        # don't need or have mlists, i.e., when eval
        if self.mlists_load_fold == "": return subframe 

        # mlist
        submlist = self.mlist
        submlist = submlist[submlist[:, 1] >= h * 32]
        submlist = submlist[submlist[:, 1] <= h * 32 + 39]
        submlist = submlist[submlist[:, 2] >= w * 32]
        submlist = submlist[submlist[:, 2] <= w * 32 + 39]
        submlist = submlist - Tensor([0, h*32, w*32])
        submlist = (submlist + 0.5) * self.up_sample - 0.5
        submlist = torch.round(submlist).int()

        # label
        sublabel = torch.zeros(*self.dim_label.tolist())
        sublabel[tuple(submlist.t())] = 1
        sublabel *= subframe  # add brightness information to label
        sublabel = torch.clip(sublabel, 0, 1)

        return subframe, sublabel

    def __len__(self) -> int:
        """
        Return: 
            self.num (int): Total number of data.
        """
        return self.num

    def _getAveragemax(self) -> float:
        """
        Since the max value of each frames is different where the brightest info
        of each frames is important, we can not use each frame's max value to
        normalize the frame. Instead, we use the average of all frames' max
        value to normalize the frame so that the brightest info of each frames
        can be preserved.
        This function will be called when initialize the dataset.
        """
        averagemax = 0
        for index in range(len(self.frames_list)//10):
            averagemax += torch.from_numpy(tifffile.imread(
                os.path.join(self.frames_load_fold, self.frames_list[index])
            )).max().float()
        averagemax /= len(self.frames_list)
        return averagemax

    def _readNext(self, index: int, pad: int = 4) -> None:
        # frame
        self.frame = torch.from_numpy(tifffile.imread(
            os.path.join(self.frames_load_fold, self.frames_list[index])
        ))
        self.frame = (self.frame / self.averagemax).float()
        self.frame = F.interpolate(
            self.frame.unsqueeze(0).unsqueeze(0), 
            size = (32, 512, 512)
        ).squeeze(0).squeeze(0)
        self.frame = F.pad(self.frame, (pad, pad, pad, pad, pad, pad))

        # don't need or have mlists, i.e., when eval
        if self.mlists_load_fold == "": return
        
        # mlist
        _, self.mlist = scipy.io.loadmat(
            os.path.join(self.mlists_load_fold, self.mlists_list[index])
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
            dataset = _SimDataset(config, config.num[d])
        elif config.type_data[d] == "Raw":
            dataset = _RawDataset(config, config.num[d])
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
    from tifffile import imsave

    data_save_fold = "test"
    if not os.path.exists(data_save_fold): os.makedirs(data_save_fold)

    # test the SimDataset
    dataset = _SimDataset(
        num = 1, 
        dim_nm_src = [130, 130, 130], 
        dim_px_dst = [160, 160, 160], 
        scale_list = [2, 4, 8, 16], 
        std_nm_src =  [
            [130, 130, 130],
            [320, 260, 260],
        ]
    )
    frame, label = dataset[0]
    print("Number of molecular:", len(torch.nonzero(label)))
    imsave(data_save_fold + '/SimFrame.tif', frame.numpy())
    imsave(data_save_fold + '/SimLabel.tif', label.numpy())

    # test the RawDataset
    #dataset = _RawDataset(config, 5000)
    #frame, label = dataset[0]
    #imsave(data_save_fold + '/RawFrame.tif', frame.numpy())
    #imsave(data_save_fold + '/RawLabel.tif', label.numpy())

    # density of RawDataset
    #num = 0
    #for i in range(5000):
    #    frame, label = dataset[i]
    #    num += len(torch.nonzero(label))
    #print("density of RawDataset:", num / 5000)
