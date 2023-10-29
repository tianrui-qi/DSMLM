import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import Tensor

import os
import tifffile
import h5py
import scipy.io
import tqdm
from typing import Tuple, Union

import sml.config


class SimDataset(Dataset):
    def __init__(self, config: sml.config.TrainerConfig, num: int) -> None:
        super(SimDataset, self).__init__()
        self.num = num

        # luminance/brightness information
        self.lum_info = config.lum_info
        # dimension
        self.D = len(config.dim_dst)                    # # of dimension 
        self.dim_src = None                             # [D], int
        self.dim_dst = Tensor(config.dim_dst).int()     # [D], int
        # scale up factor 
        self.scale_list = Tensor(config.scale_list).int()          
        self.scale      = None                          # [D], int
        # molecular profile
        self.std_src = Tensor(config.std_src)           # [2, D], float

        # store molecular list for current frame
        self.N        = None    # # of molecula    
        self.mean_set = None    # [N, D], float
        self.vars_set = None    # [N, D], float
        self.peak_set = None    # [N], float   

    # TODO: how to add brightness info, now we use peak
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        # generate molecular list for current frame
        self._generateMlist()

        frame = torch.zeros(self.dim_src.tolist())
        label = torch.zeros(self.dim_dst.tolist())
        for m in range(self.N):
            ## mlist
            mean = self.mean_set[m]     # [D], float
            vars = self.vars_set[m]     # [D], float
            peak = self.peak_set[m]     # float

            ## frame
            # take a slice around the mean where the radia is std*4
            ra = torch.ceil(torch.sqrt(vars)*4).int()   # radius, [D]
            lo = torch.maximum(torch.round(mean)-ra, torch.zeros(self.D)).int()
            up = torch.minimum(torch.round(mean)+ra, self.dim_src-1).int()
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
            # dim_src -> dim_dst
            mean = torch.round((mean + 0.5) * self.scale - 0.5)
            # set the brightness to the peak of gaussian/molecular
            label[tuple(mean.int())] += peak if self.lum_info else 1

        # add noise
        save_bit = 2**(3+torch.randint(0, 3, (1,)))     # 8, 16, or 32 bit
        frame = self._generateNoise(frame, save_bit=save_bit)
        # dim_src -> dim_dst
        frame = F.interpolate(
            frame.unsqueeze(0).unsqueeze(0),
            scale_factor=self.scale.tolist()
        ).squeeze(0).squeeze(0)

        return torch.clip(frame, 0, 1), torch.clip(label, 0, 1)

    def _generateMlist(self) -> None:
        # random scale up factor
        self.scale = torch.randint(0, len(self.scale_list), (self.D,))
        self.scale = self.scale_list[self.scale]            # [D], int

        # pixel number of source frame
        self.dim_src = (self.dim_dst / self.scale).int()    # [D], int

        # number of molecular
        self.N = torch.sum(self.dim_src).int()      # max possible # of mol
        self.N = torch.randint(0, self.N+1, (1,))   # random mol # from 0 to max

        # generate parameters in source dimension
        # moleculars' mean, [N, D]
        self.mean_set  = torch.rand(self.N, self.D) * (self.dim_src - 1)
        # moleculars' variance, [N, D]
        self.vars_set  = torch.rand(self.N, self.D)
        self.vars_set *= self.std_src[1] - self.std_src[0]
        self.vars_set += self.std_src[0]
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

    def __len__(self) -> int:
        """
        Return: 
            self.num (int): Total number of data.
        """
        return self.num


class RawDataset(Dataset):
    def __init__(self, config: sml.config.Config, num: int) -> None:
        super(RawDataset, self).__init__()
        self.num  = num
        self.mode = None
        if isinstance(config, sml.config.TrainerConfig): self.mode = "train"
        if isinstance(config, sml.config.EvaluerConfig): self.mode = "evalu"

        # luminance/brightness information
        self.lum_info = config.lum_info
        # dimension
        self.D = len(config.dim_dst)   # # of dimension
        self.dim_src         = None
        self.dim_src_pad     = None
        self.dim_src_raw     = None
        self.dim_src_raw_pad = None
        self.dim_dst         = None
        self.dim_dst_pad = Tensor(config.dim_dst).int()     # [D], int
        self.dim_dst_raw     = None
        self.dim_dst_raw_pad = None
        # scale up factor
        self.scale   = Tensor(config.scale).int()   # [D], int
        # pad for patching
        self.pad_src = Tensor([2, 2, 2]).int()      # [D], int

        # subframe index
        self.num_sub      = None    # [D], int
        self.num_sub_user = None    # [D], int
        self.rng_sub_user = None    # [D, 2], int

        # data path and file name list
        self.frames_load_fold = config.frames_load_fold
        self.frames_list = os.listdir(self.frames_load_fold)
        if self.mode == "train":
            self.mlists_load_fold = config.mlists_load_fold
            self.mlists_list = os.listdir(self.mlists_load_fold)
        # read option
        self.averagemax = None  # for normalizing all frames

        # store the current frame and mlist in memory
        self.current_frame_index = -1
        self.frame = None
        self.mlist = None   # [N, 7], float  

        self._getIndex()
        self._getAveragemax()

    def _getIndex(self) -> None:
        # compute dimension of patch
        self.dim_src = (self.dim_dst_pad / self.scale).int() - 2*self.pad_src   
        self.dim_src_pad = self.dim_src + 2*self.pad_src 
        self.dim_src_raw = torch.tensor(torch.from_numpy(tifffile.imread(
                os.path.join(self.frames_load_fold, self.frames_list[0])
        )).shape).int()
        self.num_sub = torch.ceil(self.dim_src_raw / self.dim_src).int()
        self.dim_src_raw_pad = self.num_sub * self.dim_src + 2*self.pad_src
        self.dim_dst         = self.scale * self.dim_src 
        self.dim_dst_pad     = self.dim_dst_pad
        self.dim_dst_raw     = self.scale * self.dim_src_raw
        self.dim_dst_raw_pad = self.scale * self.dim_src_raw_pad

        ## train

        if self.mode == "train":
            self.rng_sub_user = Tensor([
                [0, self.num_sub[0]],
                [0, self.num_sub[1]],
                [0, self.num_sub[2]]
            ]).int()
            self.num_sub_user = self.rng_sub_user[:, 1]
            # self.num must be given when instantiate in train mode
            return

        ## evalu

        # draw all patch
        patch = torch.zeros(self.dim_src_raw_pad.tolist())
        patch[
            self.pad_src[0] : self.pad_src[0] + self.dim_src_raw[0],
            self.pad_src[1] : self.pad_src[1] + self.dim_src_raw[1],
            self.pad_src[2] : self.pad_src[2] + self.dim_src_raw[2],
        ] = 1
        for c in range(self.num_sub[0]+1):
            index = c * self.dim_src[0] + (self.pad_src[0]-1)
            patch[index : index+2, :, :] = 0.1
        for h in range(self.num_sub[1]+1):
            index = h * self.dim_src[1] + (self.pad_src[1]-1)
            patch[:, index : index+2, :] = 0.1
        for w in range(self.num_sub[2]+1):
            index = w * self.dim_src[2] + (self.pad_src[2]-1)
            patch[:, :, index : index+2] = 0.1
        if not os.path.exists("data"): os.makedirs("data")
        tifffile.imwrite("data/patch.tif", patch.numpy())
        print("Check data/patch.tif for the all patch.")

        # ask user to input sub_range, six number
        print("Number of subframe: ", self.num_sub.tolist())
        print("Type the subframe start, end index for each dimension:")
        input_string = input()
        self.rng_sub_user = [
            0, self.num_sub[0], 0, self.num_sub[1], 0, self.num_sub[2]
        ] if input_string == "" else [
            int(item.strip()) for item in input_string.split(",")
        ]
        self.rng_sub_user = Tensor(self.rng_sub_user).int().reshape(self.D, 2)
        self.num_sub_user = self.rng_sub_user[:, 1] - self.rng_sub_user[:, 0]
        if self.num is None:
            # self.num is not necessary when instantiate in evalu mode
            # if not given, compute it automatically 
            self.num = torch.prod(self.num_sub_user) * len(self.frames_list)

        # draw selected patch
        patch = torch.zeros(self.dim_src_raw_pad.tolist())
        patch[
            self.pad_src[0] + self.dim_src[0] * self.rng_sub_user[0][0] :
            self.pad_src[0] + self.dim_src[0] * self.rng_sub_user[0][1],
            self.pad_src[1] + self.dim_src[1] * self.rng_sub_user[1][0] :
            self.pad_src[1] + self.dim_src[1] * self.rng_sub_user[1][1],
            self.pad_src[2] + self.dim_src[2] * self.rng_sub_user[2][0] :
            self.pad_src[2] + self.dim_src[2] * self.rng_sub_user[2][1],
        ] = 1
        for c in range(self.num_sub[0]+1):
            index = c * self.dim_src[0] + (self.pad_src[0]-1)
            patch[index : index+2, :, :] = 0.1
        for h in range(self.num_sub[1]+1):
            index = h * self.dim_src[1] + (self.pad_src[1]-1)
            patch[:, index : index+2, :] = 0.1
        for w in range(self.num_sub[2]+1):
            index = w * self.dim_src[2] + (self.pad_src[2]-1)
            patch[:, :, index : index+2] = 0.1
        if not os.path.exists("data"): os.makedirs("data")
        tifffile.imwrite("data/patch.tif", patch.numpy())
        print("Check data/patch.tif for selected patch.")

    def _getAveragemax(self) -> None:
        """
        Since the max value of each frames is different where the brightest info
        of each frames is important, we can not use each frame's max value to
        normalize the frame. Instead, we use the average of all frames' max
        value to normalize the frame so that the brightest info of each frames
        can be preserved.
        This function will be called when initialize the dataset.
        """
        self.averagemax = 0
        for index in tqdm.tqdm(
            range(len(self.frames_list)//10), unit="frame", leave=False
        ):
            self.averagemax += torch.from_numpy(tifffile.imread(
                os.path.join(self.frames_load_fold, self.frames_list[index])
            )).max().float()
        self.averagemax /= len(self.frames_list)

    # TODO: how to add brightness info, now we use peak
    def __getitem__(self, index: int) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        frame_index = index // torch.prod(self.num_sub_user)    # frame index
        sub_index   = index %  torch.prod(self.num_sub_user)    # subframe index
        # channel index
        c = sub_index // (self.num_sub_user[1] * self.num_sub_user[2])    
        c = c + self.rng_sub_user[0][0]
        # height index
        h = sub_index %  (self.num_sub_user[1] * self.num_sub_user[2])    
        h = h // self.num_sub_user[2] + self.rng_sub_user[1][0]
        # width index 
        w = sub_index %  self.num_sub_user[2] + self.rng_sub_user[2][0] 

        # load new frame and mlist if frame index is different
        if frame_index != self.current_frame_index: 
            self._readNext(frame_index)
            self.current_frame_index = frame_index

        # frame
        subframe = self.frame[
            c * self.dim_src[0] : c * self.dim_src[0] + self.dim_src_pad[0],
            h * self.dim_src[1] : h * self.dim_src[1] + self.dim_src_pad[1],
            w * self.dim_src[2] : w * self.dim_src[2] + self.dim_src_pad[2],
        ]
        subframe = F.interpolate(
            subframe.unsqueeze(0).unsqueeze(0), 
            scale_factor=self.scale.tolist()
        ).squeeze(0).squeeze(0)

        # don't need or have mlists, i.e., when evalu
        if self.mode == "evalu": return subframe 

        # mlist
        submlist = self.mlist
        submlist = submlist[
            submlist[:, 0] >= -0.5 + c * self.dim_src[0]
        ]
        submlist = submlist[
            submlist[:, 0] <  -0.5 + c * self.dim_src[0] + self.dim_src_pad[0]
        ]
        submlist = submlist[
            submlist[:, 1] >= -0.5 + h * self.dim_src[1]
        ]
        submlist = submlist[
            submlist[:, 1] <  -0.5 + h * self.dim_src[1] + self.dim_src_pad[1]
        ]
        submlist = submlist[
            submlist[:, 2] >= -0.5 + w * self.dim_src[2]
        ]
        submlist = submlist[
            submlist[:, 2] <  -0.5 + w * self.dim_src[2] + self.dim_src_pad[2]
        ]
        submlist[:, 0:3] = submlist[:, 0:3] - Tensor([
            c * self.dim_src[0], h * self.dim_src[1], w * self.dim_src[2]
        ])
        submlist[:, 0:3] = (submlist[:, 0:3] + 0.5) * self.scale - 0.5
        submlist[:, 0:3] = torch.round(submlist[:, 0:3])

        # label
        sublabel = torch.zeros(*self.dim_dst_pad.tolist())
        for m in range(len(submlist)):
            mean = submlist[m, 0:3]     # [D], float
            peak = submlist[m,   6]     # float
            # set the brightness to the peak of gaussian/molecular
            sublabel[tuple(mean.int())] += peak if self.lum_info else 1

        return torch.clip(subframe, 0, 1), torch.clip(sublabel, 0, 1)

    def _readNext(self, index: int) -> None:
        ## frame
        # read frame
        self.frame = torch.from_numpy(tifffile.imread(
            os.path.join(self.frames_load_fold, self.frames_list[index])
        )).float()
        # normalize
        self.frame = self.frame / self.averagemax
        # pad frame to self.dim_src_raw_pad
        right_pad  = self.dim_src_raw_pad - self.dim_src_raw - self.pad_src
        self.frame = F.pad(self.frame, (
            self.pad_src[2], right_pad[2],
            self.pad_src[1], right_pad[1],
            self.pad_src[0], right_pad[0], 
        ))

        # don't need or have mlists, i.e., when evalu
        if self.mode == "evalu": return

        ## mlist
        # read mlist
        try:
            with h5py.File(os.path.join(
                self.mlists_load_fold, self.mlists_list[index]
            ), 'r') as file: self.mlist = file['storm_coords'][()].T
        except OSError:
            _, self.mlist = scipy.io.loadmat(
                os.path.join(self.mlists_load_fold, self.mlists_list[index])
            ).popitem()
        self.mlist = torch.from_numpy(self.mlist).float()
        # normalize
        self.mlist[:,   6] /= self.averagemax
        # match the coordinate after padding
        self.mlist[:, 0:3] += self.pad_src

    def combineFrame(self, subframes: Tensor) -> Tensor:
        frame = torch.zeros(
            (self.num_sub_user * self.dim_dst).int().tolist(), 
            dtype=subframes.dtype, device=subframes.device
        )

        for sub_index in range(torch.prod(self.num_sub_user)):
            c = sub_index // (self.num_sub_user[1] * self.num_sub_user[2])
            h = sub_index %  (self.num_sub_user[1] * self.num_sub_user[2])
            h = h // self.num_sub_user[2]
            w = sub_index %  self.num_sub_user[2]

            frame[
                c * self.dim_dst[0] : (c+1) * self.dim_dst[0],
                h * self.dim_dst[1] : (h+1) * self.dim_dst[1], 
                w * self.dim_dst[2] : (w+1) * self.dim_dst[2],
            ] = subframes[
                sub_index,
                  self.pad_src[0] * self.scale[0] : 
                - self.pad_src[0] * self.scale[0],
                  self.pad_src[1] * self.scale[1] : 
                - self.pad_src[1] * self.scale[1],
                  self.pad_src[2] * self.scale[2] : 
                - self.pad_src[2] * self.scale[2],
            ]

        # right pad exceed the raw frame
        exceed_src = self.dim_src_raw_pad - self.dim_src_raw - 2*self.pad_src
        exceed_dst = exceed_src * self.scale
        if self.rng_sub_user[0][1] == self.num_sub[0]:
            frame = frame[:-exceed_dst[0], :, :]
        if self.rng_sub_user[1][1] == self.num_sub[1]:
            frame = frame[:, :-exceed_dst[1], :]
        if self.rng_sub_user[2][1] == self.num_sub[2]:
            frame = frame[:, :, :-exceed_dst[2]]

        return frame

    def __len__(self) -> int:
        """
        Return: 
            self.num (int): Total number of data.
        """
        return self.num


if __name__ == "__main__":
    config = sml.config.Config()

    data_save_fold = "temp"
    if not os.path.exists(data_save_fold): os.makedirs(data_save_fold)

    # test the SimDataset
    #"""
    dataset = SimDataset(config, num = 1)
    frame, label = dataset[0]
    print("Number of molecular:", len(torch.nonzero(label)))
    tifffile.imsave(data_save_fold + '/SimFrame.tif', frame.numpy())
    tifffile.imsave(data_save_fold + '/SimLabel.tif', label.numpy())
    #"""

    # test the RawDataset
    #"""
    dataset = RawDataset(config, num = 1, mode="train")
    frame, label = dataset[512]
    tifffile.imsave(data_save_fold + '/RawFrame.tif', frame.numpy())
    tifffile.imsave(data_save_fold + '/RawLabel.tif', label.numpy())
    #"""
