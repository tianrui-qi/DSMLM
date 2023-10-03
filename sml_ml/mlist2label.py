import torch
import torch.nn.functional as F
from torch import Tensor

import os
import tifffile
import scipy.io
import tqdm


def mlist2label(
    # dimensional config
    dim_frame, up_sample,
    # whether using luminance information
    lum_info,
    # data path
    frames_load_fold, mlists_load_fold, data_save_fold
):
    """
    Generating Frames using mlist from traditional method
    """

    # dimensional config
    dim_frame = Tensor(dim_frame).int()
    up_sample = Tensor(up_sample).int()
    dim_label = dim_frame * up_sample

    # file name list
    frames_list = os.listdir(frames_load_fold)
    mlists_list = os.listdir(mlists_load_fold)

    # create folder
    if not os.path.exists(data_save_fold): os.makedirs(data_save_fold)

    result = torch.zeros(*dim_label.tolist())
    for index in tqdm.tqdm(
        range(len(frames_list)), desc=data_save_fold, unit="frame"
    ):
        # frame
        frame = torch.from_numpy(tifffile.imread(
            os.path.join(frames_load_fold, frames_list[index])
        ))
        # we got 6.5 by averging the max value of all frames
        frame = (frame / 6.5).float()
        frame = F.interpolate(
            frame.unsqueeze(0).unsqueeze(0), 
            size = (32, 512, 512)
        ).squeeze(0).squeeze(0)
        frame = torch.clip(frame, 0, 1)

        frame = F.interpolate(
            frame.unsqueeze(0).unsqueeze(0), 
            scale_factor=up_sample.tolist()
        ).squeeze(0).squeeze(0)

        # mlist
        _, mlist = scipy.io.loadmat(
            os.path.join(mlists_load_fold, mlists_list[index])
        ).popitem()
        mlist = torch.from_numpy(mlist).float()
        mlist = mlist[:, [2, 0, 1]] - 1  # (H W D) -> (D H W)
        mlist[:, 0] = (mlist[:, 0] + 0.5) / 2 - 0.5

        mlist = (mlist + 0.5) * up_sample - 0.5
        mlist = torch.round(mlist).int()

        # label
        label = torch.zeros(*dim_label.tolist())
        label[tuple(mlist.t())] = 1
        if lum_info: label *= frame
        result += label

        # save after combine 1, 2, 4, 8, 16... frames
        current_frame = index + 1
        if current_frame & (current_frame - 1) == 0 or index == 45000 - 1:
            tifffile.imwrite(
                "{}/{:05}.tif".format(data_save_fold, current_frame),
                result.numpy()
            )


if __name__ == "__main__":
    mlist2label(
        # dimensional config
        dim_frame = [ 32, 512, 512],    # (C, H, W), (130, 130, 130)nm
        up_sample = [  4,   8,   8],    # (C, H, W)
        # whether using luminance information
        lum_info  = True,               # by label *= frame
        # data path
        frames_load_fold = "D:/frames",
        mlists_load_fold = "D:/mlists",
        data_save_fold   = "D:/temp",
    )
