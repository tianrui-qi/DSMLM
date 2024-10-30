import torch
import torch.nn.functional as F

import numpy as np
import scipy.io
import scipy.ndimage

import os
import tifffile
import tqdm
import h5py
import argparse


def main() -> None:
    args = getArgs()
    if args.mode == "preprocessPSF":    preprocessPSF(**vars(args))
    if args.mode == "preprocessFrame":  preprocessFrame(**vars(args))
    if args.mode == "preprocessMList":  preprocessMList(**vars(args))


def getArgs():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # preprocessPSF
    parser_preprocessPSF = subparsers.add_parser('preprocessPSF')
    parser_preprocessPSF.add_argument(
        '-P', '--psf_load_path', type=str, required=True, dest="psf_load_path",
    )
    parser_preprocessPSF.add_argument(
        '-O', '--psf_save_path', type=str, required=True, dest="psf_save_path",
    )
    parser_preprocessPSF.add_argument(
        '-c', '--crop', type=int, default=16, dest="crop",
    )

    # preprocessFrame
    parser_preprocessFrame = subparsers.add_parser('preprocessFrame')
    parser_preprocessFrame.add_argument(
        '-L', '--frames_load_fold', type=str, required=True, 
        dest="frames_load_fold",
    )
    parser_preprocessFrame.add_argument(
        '-S', '--frames_save_fold', type=str, required=True, 
        dest="frames_save_fold",
    )

    # preprocessMList
    parser_preprocessMList = subparsers.add_parser('preprocessMList')
    parser_preprocessMList.add_argument(
        '-L', '--mlists_load_fold', type=str, required=True, 
        dest="mlists_load_fold",
    )
    parser_preprocessMList.add_argument(
        '-S', '--mlists_save_fold', type=str, required=True, 
        dest="mlists_save_fold",
    )

    return parser.parse_args()


def preprocessPSF(
    psf_load_path: str, psf_save_path: str, crop: int = 16, **kwargs
) -> None:
    #psf_load_path = "data/psf.mat"
    #psf_save_path = "data/psf.tif"

    # (1024, 1024, 155) HWD  65nm float64
    psf = scipy.io.loadmat(psf_load_path)['FLFPSF']
    # (128, 1024, 1024) DWH  65nm float32
    psf = psf.astype('float32').transpose(2, 0, 1)[14:-13,:,:]
    # ( 64,  512,  512) DWH 130nm float32
    psf = scipy.ndimage.zoom(psf, zoom=0.5)
    # ( 32,   32,   32) DHW 130nm float32
    psf = psf[
        int(psf.shape[0]/2)-crop : int(psf.shape[0]/2)+crop,
        int(psf.shape[1]/2)-crop : int(psf.shape[1]/2)+crop,
        int(psf.shape[2]/2)-crop : int(psf.shape[2]/2)+crop,
    ]

    tifffile.imsave(psf_save_path, psf)


def preprocessFrame(
    frames_load_fold: str, frames_save_fold: str, **kwargs
) -> None:
    """
    Convert the Xunamen output 
    from (64 512 512)pixel ( 65 130 130)nm 16bit
    to   (32 512 512)pixel (130 130 130)nm 16bit
    To save storage space.
    """
    
    if not os.path.exists(frames_save_fold):
        os.makedirs(frames_save_fold)
    else:
        raise FileExistsError(
            "Folder {} already exists.".format(frames_save_fold)
        )
    
    frames_list = os.listdir(frames_load_fold)
    for index in tqdm.tqdm(range(len(frames_list))):
        # read 16 bit tiff, note that the order of os.listdir is not guaranteed
        frame = tifffile.imread(
            os.path.join(frames_load_fold, frames_list[index])
        ).astype(np.float32)
        frame = torch.from_numpy(frame).float()
        # 64 512 512 -> 32 512 512
        frame = F.interpolate(
            frame.unsqueeze(0).unsqueeze(0), 
            size = (32, 512, 512)
        ).squeeze(0).squeeze(0).half()
        # save 16 bit with file name formation
        tifffile.imwrite(
            "{}/{:05}.tif".format(
                frames_save_fold, int(frames_list[index][11:-4])
            ), frame.numpy()
        )


def preprocessMList(
    mlists_load_fold: str, mlists_save_fold: str, **kwargs
) -> None:
    """
    (x, y, z, FWHM_x, FWHM_y, FWHM_z, peak) -> 
        (z, x, y, var_z, var_x, var_y, peak)
    x, y, z start index from 1 to 0, i.e., 1-64 to 0-63
    z pixel size from 65 to 130, (64, 512, 512) -> (32, 512, 512)
    """

    mlists_list = os.listdir(mlists_load_fold)
    for index in tqdm.tqdm(range(len(mlists_list))):
        # read mlist
        try:
            with h5py.File(
                os.path.join(mlists_load_fold, mlists_list[index]), 'r'
            ) as file: mlist = file['storm_coords'][()].T
        except OSError:
            _, mlist = scipy.io.loadmat(
                os.path.join(mlists_load_fold, mlists_list[index])
            ).popitem()
        mlist = torch.from_numpy(mlist).float()

        # (x, y, z, FWHM_x, FWHM_y, FWHM_z, peak) -> 
        # (z, x, y, FWHM_z, FWHM_x, FWHM_y, peak)
        mlist = mlist[:, [2, 0, 1, 5, 3, 4, 6]]
        # index from (0 - 64) to (-0.5 - 63.5)
        mlist[:, 0:3] = mlist[:, 0:3] - 0.5
        # (65, 130, 130)nm -> (130, 130, 130)nm
        mlist[:, 0] = (mlist[:, 0] + 0.5) / 2 - 0.5
        mlist[:, 3] =  mlist[:, 3] / 2
        # (x, y, z, FWHM_x, FWHM_y, FWHM_z, peak) -> 
        # (z, x, y, var_z, var_x, var_y, peak)
        mlist[:, 3:6] = (mlist[:, 3:6] / 2.355) ** 2

        # save
        scipy.io.savemat(
            "{}/{:05}.mat".format(
                mlists_save_fold, int(mlists_list[index][11:-4])
            ), {"storm_coords": mlist.numpy()}
        )


if __name__ == "__main__": main()
