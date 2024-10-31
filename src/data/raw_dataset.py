import torch
import torch.nn.functional as F
import torch.utils.data                 # Dataset
from torch import Tensor

import os
import tqdm 
import tifffile
import h5py
import scipy.io


__all__ = ["RawDataset"]


class RawDataset(torch.utils.data.Dataset):
    def __init__(
        self, num: int | None, lum_info: bool, dim_dst: list[int], 
        scale: list[int], rng_sub_user: list[int] | None,
        frames_load_fold: str, mlists_load_fold: str | None
    ) -> None:
        super(RawDataset, self).__init__()
        self.num  = num
        self.mode = "evalu" if num is None else "train"

        # luminance/brightness information
        self.lum_info = lum_info
        # dimension
        self.D = len(dim_dst)   # # of dimension
        self.dim_src         = None
        self.dim_src_pad     = None
        self.dim_src_raw     = None
        self.dim_src_raw_pad = None
        self.dim_dst         = None
        self.dim_dst_pad = Tensor(dim_dst).int()    # [D], int
        self.dim_dst_raw     = None
        self.dim_dst_raw_pad = None
        # scale up factor
        self.scale   = Tensor(scale).int()      # [D], int
        # pad for patching
        self.pad_src = Tensor([2, 2, 2]).int()  # [D], int

        # subframe index
        self.num_sub      = None            # [D], int
        self.num_sub_user = None            # [D], int
        self.rng_sub_user = rng_sub_user    # [D, 2], int

        # data path and file name list
        self.frames_load_fold = os.path.normpath(frames_load_fold)
        self.frames_list = os.listdir(self.frames_load_fold)
        if self.mode == "train":
            self.mlists_load_fold = os.path.normpath(mlists_load_fold)
            self.mlists_list = os.listdir(self.mlists_load_fold)
        # read option
        self.averagemax = None      # for normalizing all frames

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

        if self.rng_sub_user is None:
            # ask user to input sub_range, six number
            input_string = input(
                "The number of subframe is " + 
                "{} in (Z, Y, X). ".format(self.num_sub.tolist()) + 
                "Please type six int separated by comma " + 
                "as the subframe start (inclusive) and end (exclusive) " + 
                "index for each dimension, i.e., '0, 1, 8, 12, 9, 13': "
            )
            self.rng_sub_user = [
                0, self.num_sub[0], 0, self.num_sub[1], 0, self.num_sub[2]
            ] if input_string == "" else [
                int(item.strip()) for item in input_string.split(",")
            ]
        self.rng_sub_user = Tensor(self.rng_sub_user).int().reshape(self.D, 2)
        self.num_sub_user = self.rng_sub_user[:, 1] - self.rng_sub_user[:, 0]
        # compute total number of data automatically
        self.num = torch.prod(self.num_sub_user) * len(self.frames_list)

    def _getAveragemax(self) -> None:
        """
        Since the max value of each frames is different where the brightest info
        of each frames is important, we can not use each frame's max value to
        normalize the frame. Instead, we use the average of all frames' max
        value to normalize the frame so that the brightest info of each frames
        can be preserved.
        This function will be called when initialize the dataset.

        Return:
            self.averagemax (float): Average of all frames' max value.
        """
        self.averagemax = 0
        for index in tqdm.tqdm(
            range(len(self.frames_list)//10), leave=False,
            unit="frame", desc="_getAveragemax", smoothing=0.0,
        ):
            self.averagemax += torch.from_numpy(tifffile.imread(
                os.path.join(self.frames_load_fold, self.frames_list[index])
            )).float().max()
        self.averagemax /= len(self.frames_list)//10
        return self.averagemax

    def __getitem__(self, index: int) -> Tensor | tuple[Tensor, Tensor]:
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
        if self.rng_sub_user[0][1] == self.num_sub[0] and exceed_dst[0] > 0:
            frame = frame[:-exceed_dst[0], :, :]
        if self.rng_sub_user[1][1] == self.num_sub[1] and exceed_dst[1] > 0:
            frame = frame[:, :-exceed_dst[1], :]
        if self.rng_sub_user[2][1] == self.num_sub[2] and exceed_dst[2] > 0:
            frame = frame[:, :, :-exceed_dst[2]]

        return frame

    def __len__(self) -> int:
        """
        Return: 
            self.num (int): Total number of data.
        """
        return self.num
