import os
import scipy.io as sio
from tifffile import imsave, imread
from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.distributions.multivariate_normal import MultivariateNormal


__all__ = [
    "SimuDataset", "SimuDataLoader",  # Simulated data
    "CropDataset", "CropDataLoader",  # Crop data from raw data
]


# ============================================================================ #
#                                Simulated data                                #
# ============================================================================ #


class SimuDataset(Dataset):
    def __init__(self, config) -> None:
        super(SimuDataset, self).__init__()

        ## Configuration (final)
        # dimensional config
        self.dim_frame = Tensor(config.dim_frame).int()     # [D]
        self.up_sample = Tensor(config.up_sample).int()     # [D]
        self.dim_label = self.dim_frame * self.up_sample    # [D]
        # number of data
        self.num       = config.num
        # config for adjust distribution of molecular
        self.mol_epoch = config.mol_epoch
        self.mol_range = config.mol_range   # [2]
        self.std_range = config.std_range   # [2]
        self.lum_range = config.lum_range   # [2]
        # config for reducing resolution and adding noise
        self.bitdepth  = config.bitdepth
        self.qe        = config.qe
        self.sen       = config.sen
        self.noise_mu  = config.noise_mu
        self.noise_var = config.noise_var

        ## Index (final)
        self.D   = len(self.dim_frame)  # number of dimension
        self.rad = torch.ceil(3 * self.std_range[1] * self.up_sample).int()    
        self.dai = 2 * self.rad + 1     # [D], diameter

        ## Variable (dynamic)
        self.mol_list = torch.empty(self.mol_epoch, *self.dai.tolist())

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        We first use the help function to generate a frame and label that super
        resoluted. Then, we reduce the resolution and add noise to the frame
        using help function `generateNoise`. Note that before calling this
        function, we have to call `generateMlist` to generate molecular list 
        first.

        Args:
            idx (int): Index of the data. Since we random generate data, this
                index is not used.
        
        Return:
            frame (Tensor): [*self.dim_frame], low resolution
            label (Tensor): [*self.dim_label], super resolution
        """
        frame, label = self.generateData()
        frame = self.generateNoise(frame)
        return frame, label

    def __len__(self) -> int:
        """
        Return: 
            self.num (int): Total number of data, including train and valid.
        """
        return sum(self.num)

    # help functions for generate molecular list before epoch start

    def generateMlist(self) -> Tensor:
        """
        This is a help function for generat molecular list before epoch start.
        We will call this function before each epoch start in __iter__ in
        DataLoader.

        As the `self.mol_range` increase, i.e., the min and max number of
        molecular in each frame, the computational time increase dramatically
        since we have to compute the probability density for each molecular in
        each frame. To save computational time, we do not want to generate 
        molecular for each single frame. Instead, we generate `self.mol_epoch` 
        number of molecular before epoch begin where they have random standard
        deviation in each dimension, and when generate frame, we randomly choose
        a number of molecular from this list and put them in the frame with 
        random mean and luminance. Then, we have a simulated dataset that random
        enough for training where it's super fast to generate each frame.

        The molecular generated are super-resoluted. We store each molecular in
        a [*self.dai] tensor where `self.dai` is the length (diameter) of the 
        matrix in each dimension. The center of the molecular is at the center
        of the matrix, and we compute the probability density for each 
        molecular around the center with a radius of `self.rad`.

        Return:
            self.mol_list (Tensor): molecular list with shape
                [self.mol_epoch, *self.dai]
        """

        for m in range(self.mol_epoch):
            # generate variance for this molecular
            var  = torch.rand(len(self.up_sample))
            var *= self.std_range[1] - self.std_range[0]
            var += self.std_range[0]
            var  = (var * self.up_sample) ** 2

            # build coordinate system of this molecular
            index = [torch.arange(int(end)) for end in self.dai]
            grid_cell = torch.meshgrid(*index, indexing='ij')
            coord = torch.stack([c.reshape(-1) for c in grid_cell], dim=1)

            # compute the probability density for each point/pixel
            distribution = MultivariateNormal(self.rad, torch.diag(var))
            pdf  = torch.exp(distribution.log_prob(coord))
            pdf /= torch.max(pdf)  # normalized

            self.mol_list[m] = pdf.reshape(*self.dai.tolist())
        
        return self.mol_list

    # help functions for __getitem__

    def generateData(self) -> Tuple[Tensor, Tensor]:
        # number of molecular in this frame
        N = torch.randint(
            self.mol_range[0], self.mol_range[1] + 1, (1,))

        frame = torch.zeros(self.dim_label.tolist())
        label = torch.zeros(self.dim_label.tolist())
        for _ in range(N):
            # random choose a molecular
            mol = self.mol_list[torch.randint(self.mol_epoch, (1, ))]
            mol = mol.reshape(*self.dai.tolist())

            # set the luminance
            lum  = torch.rand(1) * (self.lum_range[1] - self.lum_range[0])
            lum += self.lum_range[0]
            mol *= lum

            # put the molecular in a random position in frame/label
            mean  = (torch.rand(self.D) * (self.dim_label - 1)).int()
            lower = torch.max(mean - self.rad, torch.zeros(self.D).int())
            upper = torch.min(mean + self.rad, self.dim_label - 1)
            index_mol   = tuple(slice(l, u) for l, u in zip(
                lower - (mean - self.rad), upper - (mean - self.rad)))
            index_frame = tuple(slice(l, u) for l, u in zip(lower, upper))
            frame[index_frame] += mol[index_mol]

            # put the mean of molecular in label
            label[*mean] += 1

        # prevent lum exceeding 1 or below 0
        return torch.clip(frame, 0, 1), torch.clip(label, 0, 1)

    def generateNoise(self, frame: Tensor) -> Tensor:
        """
        This function will downsample and add camer noise to the input clean and
        super-resoluted frame. 
        
        Since we use torch.nn.functional.interpolate to downsampling the frame,
        the input frame should be 1D to 3D tensor. 
        We add three type of noise in our camera noise. First we convert the
        gray frame to photons to add shot noise. Then convert to electons to 
        add dark noise. Finally, we round the frame which simulate the loss of 
        information due to limit bitdepth when store data.

        Args:
            frame (Tensor): [W], [H, W], or [D, H, W] tensor where W, H, D are
                the width, height, and depth of the frame. The frame should be
                clean and super-resoluted.
        
        Return:
            noise (Tensor): [W'], [H', W'], or [D', H', W'] tensor where W, H, D
                are the width, height, and depth of the frame. The frame is
                downsampled and added noise. Note that W', H', D' are different
                from W, H, D since we downsample the frame. The downsampling
                scale, i.e., W/W', H/H', D/D', is determined by 
                `self.up_sample`.
        """
        if frame.dim() not in [1, 2, 3]:
            raise ValueError("frame should be 1D to 3D tensor.")

        # downsampling: decrease pixel number / increase pixel size
        frame = F.interpolate(
            frame.reshape(1, 1, *self.dim_label.tolist()), 
            size=self.dim_frame.tolist(), mode="nearest"
        ).reshape(*self.dim_frame.tolist())

        frame *= 2**self.bitdepth - 1  # clean    -> gray
        frame /= self.sen              # gray     -> electons
        frame /= self.qe               # electons -> photons
        # shot noise / poisson noise
        frame  = torch.poisson(frame)
        frame *= self.qe               # photons  -> electons
        frame *= self.sen              # electons -> gray
        # dark noise / gaussian noise
        frame += torch.normal(
            self.noise_mu  * torch.rand(frame.shape),
            self.noise_var * torch.rand(frame.shape))
        # reducing resolution casue by limit bitdepth when store data
        frame  = torch.round(frame)
        frame /= 2**self.bitdepth - 1  # gray     -> noised

        return torch.clip(frame, 0, 1)  # prevent lum exceeding 1 or below 0


class SimuDataLoader(DataLoader):
    def __init__(self, config, dataset: Union[SimuDataset, Subset]) -> None:
        super().__init__(
            dataset,
            batch_size=config.batch_size, 
            num_workers=config.num_workers, 
            pin_memory=True
        )

    def __iter__(self):
        """
        Before each epoch, we generate a new molecular list using the help
        function `generateMlist` in SimuDataset. Note that we have to check if
        the dataset is SimuDataset or Subset of SimuDataset since we may use
        random split to split the dataset into train and validation set.

        Return:
            Please check the document of DataLoader of PyTorch.
        """
        if isinstance(self.dataset, SimuDataset):
            self.dataset.generateMlist()
        elif not isinstance(self.dataset.dataset, SimuDataset):  # type: ignore
            raise TypeError("dataset should be SimuDataset.")
        else:
            self.dataset.dataset.generateMlist()  # type: ignore
        return super().__iter__()


# ============================================================================ #
#                           Crop data from raw data                            #
# ============================================================================ #


class CropDataset(Dataset):
    def __init__(self, config) -> None:
        super(CropDataset, self).__init__()

        ## Configuration (final)
        # dimensional config
        self.dim_frame = Tensor(config.dim_frame).int()     # [D]
        self.up_sample = Tensor(config.up_sample).int()     # [D]
        self.dim_label = self.dim_frame * self.up_sample    # [D]
        # number of data
        self.num       = config.num
        # folder for raw data
        self.raw_folder = config.raw_folder
        self.raw_frames_folder = os.path.join(self.raw_folder, "frames")
        self.raw_labels_folder = os.path.join(self.raw_folder, "labels")
        # folder for crop data
        self.crop_folder = os.path.join(
            os.path.dirname(config.raw_folder), 
            "{}-{}".format(config.dim_frame, config.up_sample)
        )
        self.crop_frames_folder = os.path.join(self.crop_folder, "frames")
        self.crop_labels_folder = os.path.join(self.crop_folder, "labels")
        
        self.file_check()

    # TODO: implement __getitem__
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        pass

    def __len__(self) -> int:
        """
        Return: 
            self.num (int): Total number of data, including train and valid.
        """
        return sum(self.num)

    # help function for __init__

    def file_check(self) -> None:
        # check raw data
        if not os.path.exists(self.raw_folder):
            raise FileNotFoundError("raw_folder does not exist.")
        
        # check crop data
        # if data fold does not exist, mean the data has not been cropped
        if not os.path.exists(self.crop_folder): 
            os.makedirs(self.crop_frames_folder)
            os.makedirs(self.crop_labels_folder)
            self.prepareCropData()

    # TODO: change prepareCropData to universal function
    def prepareCropData(self) -> None:
        # def the function for reading files in a directory
        def get_files_in_dir(directory: str):
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_file():
                        yield entry.name
        
        # read each frame in the raw data folder
        for file_name in get_files_in_dir(self.raw_frames_folder):
            name = os.path.splitext(file_name)[0]
            
            # frame (numpy, uint8, low resolution) [D H W]
            frame = imread(os.path.join(self.raw_frames_folder, name+".tif"))
            # corresponding molecular list (numpy, float64, low resolution)
            # [H, W, D], [1, 512]
            mlist = sio.loadmat( # type: ignore
                os.path.join(self.raw_labels_folder, name+".mat")
            )["storm_coords"]
            # [1, 512] -> [0, 511]
            mlist = mlist - 1 
            # [H, W, D] -> [D, H, W]
            h, w, d = mlist[:, 0].copy(), mlist[:, 1].copy(), mlist[:, 2].copy()
            mlist[:, 0], mlist[:, 1], mlist[:, 2] = d, h, w
            
            # crop the frame to (64 64 64) with step size 24
            for sub in range(100):
                h = sub // 10  # index for height
                w = sub %  10  # index for width

                # store the cropped subframe
                subframe = frame[:, 232+h*24 : 232+h*24+64, w*24 : w*24+64]
                imsave(os.path.join(
                    self.crop_frames_folder, name+"-{}-{}.tif".format(h, w)
                ), subframe)

                # get the corresponding molecular list
                submlist = mlist.copy()
                submlist = submlist[submlist[:, 1] >= 232+h*24]
                submlist = submlist[submlist[:, 1] <  232+h*24+64]
                submlist = submlist[submlist[:, 2] >= w*24]
                submlist = submlist[submlist[:, 2] <  w*24+64]
                submlist = submlist - [0, 232+h*24, w*24]  # minus start point
                # low resolution -> high resolution, float -> int
                submlist /= np.array(frame.shape) - 1
                submlist *= np.array(frame.shape) * self.up_sample - 1
                submlist  = np.round(submlist).astype(int)
                # store the cropped submlist
                sio.savemat(os.path.join(  # type: ignore
                    self.crop_labels_folder, name+"-{}-{}.mat".format(h, w)
                ), {"storm_coords": submlist})

class CropDataLoader(DataLoader):
    # TODO: implement dataloader for real data
    pass


# ============================================================================ #
#                                    Main                                      #
# ============================================================================ #


if __name__ == "__main__":
    from tifffile import imsave
    from config import Config

    # create dir to store test file
    if not os.path.exists("assets/dataset"): os.makedirs("assets/dataset")

    # test SimuDataset using default config
    config  = Config()
    dataset = SimuDataset(config)

    prepareCropData(config)

    #"""
    # test function generateMlist
    mol_list = dataset.generateMlist()
    imsave('assets/dataset/mol.tif', 
           (mol_list[0] * 255).to(torch.uint8).numpy())
    
    # test function generateData
    frame, label = dataset.generateData()
    imsave('assets/dataset/frame.tif', (frame * 255).to(torch.uint8).numpy())
    imsave('assets/dataset/label.tif', (label * 255).to(torch.uint8).numpy())
    
    # test function generateNoise
    noise = dataset.generateNoise(frame)
    imsave('assets/dataset/noise.tif', (noise * 255).to(torch.uint8).numpy())

    # test the dataloader output
    frame, label = dataset[0]
    imsave('assets/dataset/frame0.tif', (frame * 255).to(torch.uint8).numpy())
    imsave('assets/dataset/label0.tif', (label * 255).to(torch.uint8).numpy())
    
    # test the input of network
    from torch import nn
    upsample = nn.Upsample(scale_factor=tuple(config.up_sample), mode='nearest')
    intput = upsample(frame.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    imsave('assets/dataset/intput.tif', (intput * 255).to(torch.uint8).numpy())
    #"""
