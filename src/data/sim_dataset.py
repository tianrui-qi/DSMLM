import torch
import torch.nn.functional as F
import torch.utils.data                 # Dataset
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import Tensor

import cupy as cp
import cupyx.scipy.ndimage

import tifffile


__all__ = ["SimDataset"]


class SimDataset(torch.utils.data.Dataset):
    def __init__(
        self, num: int, lum_info: bool, dim_dst: list[int], 
        scale_list: list[int], std_src: list[list[float]], psf_load_path: str,
    ) -> None:
        super(SimDataset, self).__init__()
        self.num = num

        # luminance/brightness information
        self.lum_info = lum_info
        # dimension
        self.D = len(dim_dst)                   # # of dimension 
        self.dim_src = None                     # [D], int          [random]
        self.dim_dst = Tensor(dim_dst).int()    # [D], int
        # scale up factor 
        self.scale_list = Tensor(scale_list).int()          
        self.scale      = None                  # [D], int          [random]
        # molecular profile
        self.std_src = Tensor(std_src)          # [2, D], float

        # psf for convolution
        if psf_load_path == "": 
            self.psf_src = None
        else:
            self.psf_src = cp.array(
                tifffile.imread(psf_load_path)
            ).astype(cp.float32)
            self.psf_src /= cp.sum(self.psf_src)    # normalize by sumation

        # store molecular list for current frame
        self.N       : Tensor = None            # [1, ], int        [random]
        self.mean_set: Tensor = None            # [N, D], float     [random]
        self.vars_set: Tensor = None            # [N, D], float     [random]
        self.peak_set: Tensor = None            # [N], float        [random]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
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

        # convolve frame with wide field psf
        frame = torch.from_numpy(
            cupyx.scipy.ndimage.convolve(cp.array(frame), self.psf_src).get()
        ).float() if self.psf_src is not None else frame
        # normalization
        # necessary, range of frame may exceed 1 after put all slice beck to 
        # whole frame or uncertain after convolve
        max = torch.max(frame)
        if max > 0: frame, label = frame / max, label / max
        # random set a universal luminate to frame
        # differ from the lum of molecular that let each molecular to have
        # different relative luminate; this universal luminate is necessary 
        # since normalization cause every frame have same max, which is bad
        # for generalization.
        peak = torch.rand(1)
        frame, label = frame * peak, label * peak

        # add noise to frame
        save_bit = 2**(3+torch.randint(0, 3, (1,)))     # 8, 16, or 32 bit
        frame = self.generateNoise(frame, save_bit=save_bit)
        # dim_src -> dim_dst
        frame = F.interpolate(
            frame.unsqueeze(0).unsqueeze(0),
            scale_factor=self.scale.tolist()
        ).squeeze(0).squeeze(0)

        return frame, label

    def _generateMlist(self) -> None:
        # random scale up factor
        self.scale = torch.randint(0, len(self.scale_list), (self.D,))
        self.scale = self.scale_list[self.scale]            # [D], int
        self.scale[2] = self.scale[1]  # same scale in XY

        # pixel number of source frame
        self.dim_src = (self.dim_dst / self.scale).int()    # [D], int

        # number of molecular
        self.N = torch.sum(self.dim_src).int()  # max possible mol num
        self.N = torch.randint(0, self.N+1, (1,)).int()     # [1] int

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

    @staticmethod
    def generateNoise(
        frame: Tensor, save_bit: int = 16, camera_bit: int = 16, 
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
