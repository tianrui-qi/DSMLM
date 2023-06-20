from typing import Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from skimage.transform import resize


class SimDataLoader(DataLoader):
    def __init__(self, config, num):
        super().__init__(
            SimDataset(config, num), 
            batch_size=config.batch_size, 
            num_workers=config.num_workers, 
            pin_memory=True)

    def __iter__(self):
        self.dataset.generateMlist() # type: ignore
        return super().__iter__()


class SimDataset(Dataset):
    def __init__(self, config, num: int) -> None:
        super(SimDataset, self).__init__()

        ## Configuration (final)
        # dimensional config that need to consider memory
        self.dim_frame = Tensor(config.dim_frame).int()                 # [D]
        self.up_sample = Tensor(config.up_sample).int()                 # [D]
        self.dim_label = Tensor(self.dim_frame * self.up_sample).int()  # [D]
        self.mol_epoch = int(config.mol_epoch)
        # config for adjust distribution of molecular
        self.mol_range = Tensor(config.mol_range).int()     # [2]
        self.std_range = Tensor(config.std_range)           # [2]
        self.lum_range = Tensor(config.lum_range)           # [2]
        # config for reducing resolution and adding noise
        self.bitdepth  = int(config.bitdepth)
        self.qe        = float(config.qe)
        self.sen       = float(config.sen)
        self.noise_mu  = float(config.noise_mu)
        self.noise_var = float(config.noise_var)

        ## Index (final)
        self.num = num                     # number of data each epoch
        self.D   = len(self.dim_frame)     # number of dimension
        self.rad = torch.ceil(3 * self.std_range[1] * self.up_sample).int()    
        self.dai = 2 * self.rad + 1             # [D]
        
        ## Variable (dynamic)
        self.mol_list = torch.zeros(self.mol_epoch, *self.dai.tolist())

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor]:
        frame, label = self.generateData()
        frame = self.generateNoise(frame)
        return frame, label

    def __len__(self) -> int:
        return self.num
    
    # help functions for generate molecular list before epoch start

    def generateMlist(self) -> Tensor:
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
            int(self.mol_range[0]), int(self.mol_range[1] + 1), (1,))

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
        # decrease pixel number / increase pixel size
        frame  = Tensor(resize(frame.numpy(), self.dim_frame))

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


if __name__ == "__main__":
    import os
    from tifffile import imsave
    from config import Config

    # create dir to store test file
    if not os.path.exists("assets/dataset"): os.makedirs("assets/dataset")

    # test SimDataset using default config
    config  = Config()
    dataset = SimDataset(config, 1)

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
