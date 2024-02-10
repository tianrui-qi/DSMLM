import numpy as np
import cupy as cp
import scipy.optimize
import scipy.interpolate
from numpy import ndarray

import os
import tifffile
import matplotlib.pyplot as plt
import tqdm
from typing import Tuple

__all__ = []


class DriftCorrector:
    def __init__(
        self, temp_save_fold: str, window: int, method: str = "MCC",
    ) -> None:
        # path
        self.temp_save_fold = temp_save_fold

        # parameters
        self.total, self.stride = self._getIndex()
        self.window = window
        self.window_num = (self.total-self.window)//self.stride+1
        self.crop = [32, 64, 64]
        self.method = method    # DCC, MCC, or RCC

        # result
        self.index_src = self.window/2 + np.arange(self.window_num)*self.stride
        self.index_dst = np.arange(self.total) + 1
        self.drift_src = None
        self.drift_dst = None

        # preload all images to memory
        # note that we already perform the fold check in self._getIndex()
        self.image_list = [
            os.path.join(self.temp_save_fold, "{}".format(file))
            for file in os.listdir(self.temp_save_fold) if file.endswith('.tif')
        ]

    def _getIndex(self) -> Tuple[int, int]:
        # error for no result saving in self.temp_save_fold for drift correction
        error = ValueError(
            "No result saving in temp_save_fold " +
            "`{}` for drifting correction. ".format(self.temp_save_fold) +
            "Please re-run the code and set stride not equal to 0."
        )

        # if self.temp_save_fold not exists, we must have not save the result
        if not os.path.exists(self.temp_save_fold):
            raise error
        # get list of the stride of the self.temp_save_fold
        idx = [
            int(file.split('.')[0]) 
            for file in os.listdir(self.temp_save_fold)
            if file.endswith('.tif')
        ]
        idx.sort()
        # if no .tif file found, raise error
        if not idx: raise error
        # calculate the parameters
        total = idx[-1]
        stride = idx[1]-idx[0]

        return total, stride

    def fit(self) -> ndarray:
        if os.path.exists(os.path.join(self.temp_save_fold, "drift.csv")):
            # if cache exists, load the drift from the cache
            self.drift_dst = np.loadtxt(
                os.path.join(self.temp_save_fold, "drift.csv"), delimiter=','
            )
            print(
                "Load drift from `{}`. ".format(
                    os.path.join(self.temp_save_fold, "drift.csv")
                ) + "Please delete `{}` ".format(
                    os.path.join(self.temp_save_fold, "drift.csv")
                ) + 
                "before running if you want to re-calculate the drift " + 
                "for same dataset with new window size or method. " + 
                "Please delete whole `{}` ".format(self.temp_save_fold) + 
                "before running if you want to re-calculate the drift " + 
                "for same dataset with new stride size or for a new dataset."
            )
        else:
            # calculate the drift
            if self.method == "DCC":
                self.drift_src = self._dcc()
            elif self.method == "MCC":
                self.drift_src = self._mcc()
            elif self.method == "RCC":
                self.drift_src = self._rcc()
            else:
                raise ValueError(
                    "Method must be one of DCC, MCC, or RCC, " + 
                    "but got {}".format(self.method)
                )
            self.drift_dst = self._interpolation()
            # save the drift as .csv for future use
            np.savetxt(
                os.path.join(self.temp_save_fold, "drift.csv"), 
                self.drift_dst, delimiter=','
            )
        self._plot()
        return self.drift_dst

    def _dcc(self) -> ndarray:
        drift = np.zeros([self.window_num, 3])  # [window_num, 3]

        image0 = self._getWindow(0)
        for j in tqdm.tqdm(range(0, self.window_num)):
            imagej = self._getWindow(j)
            # calculate the cross correlation
            corr = self.crossCorrelation3D(image0, imagej)
            # crop the correlation from the center to reduce fitting time
            corr = corr[
                corr.shape[0]//2 - self.crop[0]//2 : 
                corr.shape[0]//2 + self.crop[0]//2, 
                corr.shape[1]//2 - self.crop[1]//2 : 
                corr.shape[1]//2 + self.crop[1]//2, 
                corr.shape[2]//2 - self.crop[2]//2 : 
                corr.shape[2]//2 + self.crop[2]//2
            ]
            # fit the correlation with a gaussian to find the drift
            try:
                # initial guess for current drift
                if j == 0:
                    # if no previous drift, set the center of the corr as
                    # the initial guess
                    p0 = np.array([
                        (corr.shape[0]-1)/2, (corr.shape[1]-1)/2,
                        (corr.shape[2]-1)/2, 1, 1, 1, 1
                    ])
                else:
                    # set previous drift as the initial guess
                    p0 = np.array([*drift[j-1], 1, 1, 1, 1])
                # set 10 pixels around the initial guess as the bounds
                bounds=(
                    (*(p0[0:3]-10), -np.inf, -np.inf, -np.inf, -np.inf),
                    (*(p0[0:3]+10),  np.inf,  np.inf,  np.inf,  np.inf)  
                )
                # fit the correlation with a gaussian to find the drift
                drift[j] = DriftCorrector.gaussianFit(
                    corr, p0=p0, bounds=bounds
                )
            except RuntimeError:
                if j == 0: 
                    drift[j] = ((np.array(self.crop)-1)/2)
                else:
                    drift[j] = drift[j-1]
                tqdm.tqdm.write(
                    "Optimal para not found for window ({},{})".format(0, j)
                )
        # since the drift of window (0,0) must be zero, the center of drift is 
        # just drift(0,0) and need to be subtracted from drift 
        drift -= drift[0]

        return drift

    def _mcc(self) -> ndarray:
        # [window_num, window_num, 3], will sum to [window_num, 3] before return
        drift = np.zeros([self.window_num, self.window_num, 3])

        for i in tqdm.tqdm(range(self.window_num)):
            imagei = self._getWindow(i)
            for j in tqdm.tqdm(range(i, self.window_num), leave=False):
                imagej = self._getWindow(j)
                # calculate the cross correlation
                corr = self.crossCorrelation3D(imagei, imagej)
                # crop the correlation from the center to reduce fitting time
                corr = corr[
                    (corr.shape[0]//2-self.crop[0]//2) : 
                    (corr.shape[0]//2-self.crop[0]//2)+self.crop[0], 
                    (corr.shape[1]//2-self.crop[1]//2) : 
                    (corr.shape[1]//2-self.crop[1]//2)+self.crop[1], 
                    (corr.shape[2]//2-self.crop[2]//2) : 
                    (corr.shape[2]//2-self.crop[2]//2)+self.crop[2]
                ]
                # fit the correlation with a gaussian to find the drift
                try:
                    # initial guess for current drift
                    if i == j:
                        # if no previous drift, set the center of the corr as
                        # the initial guess
                        p0 = np.array([
                            (corr.shape[0]-1)/2, (corr.shape[1]-1)/2,
                            (corr.shape[2]-1)/2, 1, 1, 1, 1
                        ])
                    else:
                        # set previous drift as the initial guess
                        p0 = np.array([*drift[i][j-1], 1, 1, 1, 1])
                    # set 10 pixels around the initial guess as the bounds
                    bounds=(
                        (*(p0[0:3]-10), -np.inf, -np.inf, -np.inf, -np.inf),
                        (*(p0[0:3]+10),  np.inf,  np.inf,  np.inf,  np.inf)  
                    )
                    # fit the correlation with a gaussian to find the drift
                    drift[i][j] = DriftCorrector.gaussianFit(
                        corr, p0=p0, bounds=bounds
                    )
                except RuntimeError:
                    if i == j:
                        drift[i][j] = ((np.array(self.crop)-1)/2)
                    elif i == 0:
                        drift[i][j] = drift[i][j-1]
                    else:
                        drift[i][j] = drift[i-1][j] - drift[i-1][j-1]
                    tqdm.tqdm.write(
                        "Optimal para not found for window ({},{})".format(i, j)
                    )
        # since the drift of window (i,i) must be zero, the center of 
        # drift[i][:] is drift[i][i] and need to be subtracted from drift[i][:] 
        for i in range(self.window_num):
            for j in range(i+1, self.window_num):
                drift[i][j] -= drift[i][i]
            drift[i][i] -= drift[i][i]
        # drift is symmetric, i.e., drift(i,j) = -drift(j,i)
        drift -= drift.transpose((1, 0, 2))
        # optimal drift(j) is the avereage of drift(i,j) for all i
        drift = drift.sum(axis=0) / self.window_num
        # since the drift of window (0,0) must be zero, the center of drift is 
        # just drift(0,0) and need to be subtracted from drift 
        drift -= drift[0]

        return drift

    # TODO: implement RCC
    def _rcc(self) -> ndarray:
        pass

    def _getWindow(self, index: int) -> ndarray:
        image = tifffile.imread(self.image_list[index])
        for i in range(1, self.window // self.stride):
            image += tifffile.imread(self.image_list[index+i])
        return image

    @staticmethod
    def crossCorrelation3D(image1: ndarray, image2: ndarray) -> ndarray:
        fft_image1 = cp.fft.fftn(cp.asarray(image1))
        fft_image2 = cp.fft.fftn(cp.asarray(image2))
        corr = cp.fft.ifftn(cp.multiply(fft_image1, cp.conj(fft_image2)))
        corr = cp.fft.fftshift(cp.real(corr))
        return cp.asnumpy(corr)

    @staticmethod
    def gaussianFit(
        corr: ndarray, p0: ndarray = None, 
        bounds: Tuple[Tuple[float, ...], Tuple[float, ...]] = None
    ) -> ndarray:
        popt, _ = scipy.optimize.curve_fit(
            DriftCorrector.gaussian3D, 
            xdata=np.vstack(np.indices(corr.shape).reshape(3, -1)), 
            ydata=corr.ravel(), 
            p0=p0, bounds=bounds
        )
        return popt[0:3]

    @staticmethod
    def gaussian3D(xyz, x0, y0, z0, sigma_x, sigma_y, sigma_z, amp) -> ndarray:
        return amp * np.exp(-(
            (xyz[0] - x0) ** 2 / (2 * sigma_x ** 2) +
            (xyz[1] - y0) ** 2 / (2 * sigma_y ** 2) +
            (xyz[2] - z0) ** 2 / (2 * sigma_z ** 2)
        ))

    def _interpolation(self) -> ndarray:
        interp_func_z = scipy.interpolate.interp1d(
            self.index_src, self.drift_src[:, 0], kind='linear', 
            fill_value=(self.drift_src[0, 0], self.drift_src[-1, 0]), 
            bounds_error=False
        )
        interp_func_y = scipy.interpolate.interp1d(
            self.index_src, self.drift_src[:, 1], kind='linear', 
            fill_value=(self.drift_src[0, 1], self.drift_src[-1, 1]), 
            bounds_error=False
        )
        interp_func_x = scipy.interpolate.interp1d(
            self.index_src, self.drift_src[:, 2], kind='linear', 
            fill_value=(self.drift_src[0, 2], self.drift_src[-1, 2]), 
            bounds_error=False
        )

        drift_dst = np.zeros([self.total, 3])
        drift_dst[:, 0] = interp_func_z(self.index_dst)
        drift_dst[:, 1] = interp_func_y(self.index_dst)
        drift_dst[:, 2] = interp_func_x(self.index_dst)

        return drift_dst

    def _plot(self) -> None:
        # Drift over frames
        plt.figure()
        plt.plot(self.index_dst, self.drift_dst[:, 0], label='Z')
        plt.plot(self.index_dst, self.drift_dst[:, 1], label='Y')
        plt.plot(self.index_dst, self.drift_dst[:, 2], label='X')
        plt.legend()
        plt.grid(linestyle='--')
        plt.suptitle("Drift over frames")
        plt.title("total: {}; window: {}; stride: {}; crop: {}".format(
            self.total, self.window, self.stride, self.crop
        ))
        plt.xlabel('frame')
        plt.ylabel('drift (pixel)')

        # Drift in XY plane
        plt.figure()
        cmap = plt.get_cmap('viridis')  # Define colormap
        for i in range(len(self.drift_dst) - 1):
            plt.plot(
                -self.drift_dst[i:i+2, 2], -self.drift_dst[i:i+2, 1], 
                color=cmap(i / len(self.drift_dst))  # Use colormap to set color
            )
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(linestyle='--')
        plt.suptitle("Drift in XY plane")
        plt.title("total: {}; window: {}; stride: {}; crop: {}".format(
            self.total, self.window, self.stride, self.crop
        ))
        plt.xlabel('x (pixel)')
        plt.ylabel('y (pixel)')

        # show both plots simultaneously
        plt.show()
