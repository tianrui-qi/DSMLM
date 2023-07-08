import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal

import numpy as np
import os
from tifffile import imread, imsave
from scipy.io import loadmat, savemat  # type: ignore

from typing import Tuple, List


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

    # help functions for __getitem__

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
        
        # folder for raw data
        self.raw_folder = config.raw_folder
        self.raw_frames_folder = os.path.join(self.raw_folder, "frames")
        self.raw_mlists_folder = os.path.join(self.raw_folder, "mlists")
        # folder for crop data
        self.crop_folder = os.path.join(
            os.path.dirname(config.raw_folder), "crop")
        self.crop_frames_folder = os.path.join(self.crop_folder, "frames")
        self.crop_mlists_folder = os.path.join(self.crop_folder, "mlists")
        
        self.fileCheck()

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        This function will return the index-th frame and label by loading the
        frame and molecular list from the disk. We assume the data store in
        `self.crop_folder` where frame and molecular list are stored in sub-
        folder `frames` and `mlists` respectively. The file name of frame and
        molecular list should be `index.tif` and `index.mat` respectively. Also,
        since we load .mat file as dictionary, we assume the molecular list is
        stored in the key `mlist` of the dictionary.

        {crop_folder}
        ├── frames
        │   ├── 0.tif
        │   ├── 1.tif
        │   ├── ...
        │   └── {sum(self.num) - 1}.tif
        └── mlists
            ├── 0.mat
            ├── 1.mat
            ├── ...
            └── {sum(self.num) - 1}.mat

        For the frame store in `self.crop_frames_folder`, we assume they are low
        resolution, i.e., large pixel size, with shape [*self.dim_frame] and 
        pixel value is non-normalized. The data type can be any uint type; we
        will convert the frame to float type and normalize the frame to range
        [0, 1] before output.

        For the molecular list store in `self.crop_mlists_folder`, we assume 
        they are low resolution, i.e., large pixel size, with a shape of [N, D]
        where N is the number of molecules and D is number of dimension. The 
        data store in it represent the coordinate of center of each molecule. 
        The data type can be any float type; we will convert the molecular list 
        to high resolution, i.e, small pixel size, and then round to int. Then, 
        for each location in this high resolution molecular list, we will set 
        corresponding pixel in high resolution label frame to 1 where the label
        is a tensor with shape [*self.dim_label].

        Args:
            index (int): Index of the data. 
        
        Return:
            frame (Tensor): [*self.dim_frame], normalized, low resolution
            label (Tensor): [*self.dim_label], normalized, super resolution
        """
        
        ## frame
        # ndarray, any uint, low resolution, non-normalized
        frame = imread(
            os.path.join(self.crop_frames_folder, "{}.tif".format(index))
        )
        # tensor, float, low resolution, normalized
        frame = torch.from_numpy(frame)
        frame = frame.float() / torch.iinfo(frame.dtype).max

        ## molecular list
        # ndarray, any float, low resolution
        mlist = loadmat( # type: ignore
            os.path.join(self.crop_mlists_folder, "{}.mat".format(index))
        )["mlist"]
        # tensor, float, high resolution
        mlist = Tensor(torch.from_numpy(mlist)).float()  # to tensor
        mlist = (mlist + 0.5) * self.up_sample - 0.5
        mlist = torch.round(mlist).int()

        ## label
        # tensor, float, high resolution, normalized
        label = torch.zeros(*self.dim_label.tolist())
        label[tuple(mlist.t())] = 1

        return torch.clip(frame, 0, 1), torch.clip(label, 0, 1)

    def __len__(self) -> int:
        """
        Return: 
            self.num (int): Total number of data.
        """
        return self.num

    # help function for __init__

    def fileCheck(self) -> None:
        """
        WARNING: Do not reply on this function to check the validity of the raw 
        data. Please check all implementation and documentation of this class to
        see each function's pre-condition.

        This help function will check if the raw data folder and crop data 
        folder exist. If the raw data does not exist, it will raise 
        FileNotFoundError. If the crop data does not exist, it will create the 
        crop data by calling `cropData` function.
        """

        # check if raw data exist
        if not os.path.exists(self.raw_folder):
            raise FileNotFoundError("Raw data not exist.")
        elif not os.path.exists(self.raw_frames_folder):
            raise FileNotFoundError("Raw data frames not exist.")
        elif not os.path.exists(self.raw_mlists_folder):
            raise FileNotFoundError("Raw data mlists not exist.")
        
        # check if crop data exit
        # data fold not existing means the raw data has not been cropped
        if not os.path.exists(self.crop_folder): 
            os.makedirs(self.crop_frames_folder)
            os.makedirs(self.crop_mlists_folder)
            self.cropData()
        if len(os.listdir(self.crop_frames_folder)) < self.num:
            raise FileNotFoundError("Number of crop frames not enough.")
        if len(os.listdir(self.crop_mlists_folder)) < self.num:
            raise FileNotFoundError("Number of crop mlists not enough.")

    def cropData(self) -> None:
        """
        WARNING: This function is not a universal function. It is designed for
        specific raw data. Please check all implementation and documentation of
        this class to see each function's pre-condition.

        The help function to crop the raw data will following the pre-condition
        of __getitem__ function.

        This function assume the dim_frame = [64 64 64] and the raw data's shape
        is [64 512 512]. We will pad the raw data to [64 512 512] and then crop
        the raw data to 100 number of sub-frame with shape [64 64 64]. The step
        size of the crop is 52, which means that each sub-frame will have 12
        pixels overlap with each other.

        We difine this function inside the RawDataset class because we have to
        match the logic of process and read data between this function and
        __getitem__ function.
        """

        # def the function for reading files in a directory
        def get_files_in_dir(directory: str):
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_file():
                        yield entry.name
        
        # read each frame in the raw data folder
        num_sub = 0
        for file_name in get_files_in_dir(self.raw_frames_folder):
            name = os.path.splitext(file_name)[0]
            
            # frame 
            # ndarray, uint8, low resolution, [64 512 512], [D H W], [0, 255]
            frame = imread(
                os.path.join(self.raw_frames_folder, name+".tif")
            )
            frame = np.round(
                frame / frame.max() * 255).astype(np.uint8) # type: ignore
            # pad the frame from [512 512 64] to [532 532 64]
            frame = np.pad(frame, ( (0, 0), (10, 10), (10, 10)))

            # molecular list
            # ndarray, double, low resolution, [N 3], [N (H W D)], [1, 512]
            mlist = loadmat( # type: ignore
                os.path.join(self.raw_mlists_folder, name+".mat")
            )["storm_coords"]
            # ndarray, double, lwo resolution, [N (D H W)], [0, 511]
            mlist = mlist[:, [2, 0, 1]] - 1
            # match the coordinate of frame after padding
            mlist[:, 1:]+=10
            
            # crop the frame to 100 number of dim_frame frame with step size 52
            for sub in range(100):
                h = sub // 10  # index for height
                w = sub %  10  # index for width

                # the cropped subframe
                subframe = frame[
                    0      : 64, 
                    h * 52 : 64 + h * 52, 
                    w * 52 : 64 + w * 52,
                ]
                
                # the corresponding sub molecular list
                submlist = mlist.copy()
                submlist = submlist[submlist[:, 1] >= h * 52]
                submlist = submlist[submlist[:, 1] <= h * 52 + 63]
                submlist = submlist[submlist[:, 2] >= w * 52]
                submlist = submlist[submlist[:, 2] <= w * 52 + 63]
                submlist = submlist - [0, h*52, w*52]  # [0 63]

                # save the frame and mlist
                imsave(os.path.join(
                    self.crop_frames_folder, "{}.tif".format(num_sub)
                ), subframe)
                savemat(os.path.join(  # type: ignore
                    self.crop_mlists_folder, "{}.mat".format(num_sub)
                ), {"mlist": submlist})

                num_sub+=1

    def combineFrame(self, subframes: Tensor) -> Tensor:
        """
        This function will combine the subframes into a frame. The shape of the
        subframes should be [100 64 64 64] * [1, *self.up_sample] and the shape
        of the frame will be [64 512 512] * [*self.up_sample].
        We combine these subframes follow how we crop the frame into subframes
        in function `cropData`. For each [64 64 64] subframe, we only keep the
        center [64 52 52] * [*self.up_sample] part and discard the [0 6 6] * 
        [*self.up_sample] part on each side since the network may not be able to
        handle the boundary effect. Then, since we pad the [64 512 512] frame by
        [0 10 10] on each side in function `cropData`, after we combine the 
        subframes, we will get a [64 520 520] * [*self.up_sample] instead of 
        [64 512 512] * [*self.up_sample]. Thus, we will discard the [0 4 4] * 
        [*self.up_sample] part on each side to get the final frame with shape 
        [64 512 512] * [*self.up_sample].

        Note that this function is independent from the self.up_sample, i.e.,
        can be used for any self.up_sample. The most important thing is that the
        shape of the original frame, i.e, with shape [64 512 512] in our case
        and how we cut it into subframe in function `cropData`. Note that 
        `cropData` is also independent from self.up_sample.

        Args:
            subframes (Tensor): The subframes with shape [100 64 64 64] *
                [1, *self.up_sample].
        
        Returns:
            frame (Tensor): The frame with shape [64 512 512] * 
                [*self.up_sample] where the dtype and device same to the input
                `subframes` to avoid data copy between cpu and gpu.
        """
        shape = self.up_sample * Tensor([64, 520, 520]).int()
        frame = torch.zeros(
            shape.tolist(), dtype=subframes.dtype, device=subframes.device)
        for f in range(100):
            h = f // 10
            w = f %  10
            frame[
                :, 
                h * 52 * self.up_sample[1] : (h+1) * 52 * self.up_sample[1], 
                w * 52 * self.up_sample[2] : (w+1) * 52 * self.up_sample[2],
            ] = subframes[
                f, :,
                6 * self.up_sample[1] : -6 * self.up_sample[1],
                6 * self.up_sample[2] : -6 * self.up_sample[2],
            ]
        return frame[
            :,
            4 * self.up_sample[1] : -4 * self.up_sample[1],
            4 * self.up_sample[2] : -4 * self.up_sample[2],
        ]


def getDataLoader(config) -> List[DataLoader]:
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
    return dataloader


if __name__ == "__main__":
    from config import Config

    # create dir to store test file
    if not os.path.exists("data/test"): os.makedirs("data/test")

    # test using default config
    config  = Config()

    # test the RawDataset
    dataset = RawDataset(config, 1)
    frame, label = dataset[0]
    imsave('data/test/RawFrame.tif', (frame * 255).to(torch.uint8).numpy())
    imsave('data/test/RawLabel.tif', (label * 255).to(torch.uint8).numpy())

    # test the SimDataset
    dataset = SimDataset(config, 1)
    frame, label = dataset[0]
    imsave('data/test/SimFrame.tif', (frame * 255).to(torch.uint8).numpy())
    imsave('data/test/SimLabel.tif', (label * 255).to(torch.uint8).numpy())
