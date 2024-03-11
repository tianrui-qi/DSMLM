import numpy as np
import cupy as cp
import scipy.optimize
import scipy.interpolate
from numpy import ndarray

import os
import tqdm 
import tifffile
import matplotlib.pyplot as plt

__all__ = []


class DriftCorrector:
    def __init__(
        self, temp_save_fold: str, window: int, method: str,
    ) -> None:
        # path
        self.temp_save_fold = os.path.normpath(temp_save_fold)

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

    def _getIndex(self) -> tuple[int, int]:
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
        cache_path = os.path.join(
            self.temp_save_fold, "{}.csv".format(self.method)
        )
        if os.path.exists(cache_path):
            # if cache exists, load the drift from the cache
            self.drift_dst = np.loadtxt(cache_path, delimiter=',')
            print(
                "Load drift from `{}`. ".format(cache_path) + 
                "Delete `{}` if you want to re-calculate ".format(cache_path) + 
                "the drift for same dataset with new window size. " + 
                "Delete whole `{}` if you want ".format(self.temp_save_fold) + 
                "to re-calculate the drift for same dataset with new stride " + 
                "size. " + 
                "Delete whole `{}` or specify a ".format(self.temp_save_fold) + 
                "new path (recommend) if you want to re-calculate the drift " + 
                "for different dataset."
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
            np.savetxt(cache_path, self.drift_dst, delimiter=',')
        self._plot()
        return self.drift_dst

    def _dcc(self) -> ndarray:
        drift = np.zeros([self.window_num, 3])  # [window_num, 3]
        image0 = self._getWindow(0)
        for j in tqdm.tqdm(
            range(0, self.window_num), 
            desc=os.path.join(self.temp_save_fold, "DCC.csv"), 
            dynamic_ncols=True, smoothing=0.0,
        ):
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

        return drift    # [window_num, 3]

    def _mcc(self) -> ndarray:
        drift = self._getDriftMatrix()  # [window_num, window_num, 3]

        # drift is symmetric, i.e., drift(i,j) = -drift(j,i)
        drift -= drift.transpose((1, 0, 2))
        # optimal drift(j) is the avereage of drift(i,j) for all i
        drift = drift.sum(axis=0) / self.window_num
        # since the drift of window (0,0) must be zero, the center of drift is 
        # just drift(0,0) and need to be subtracted from drift 
        drift -= drift[0]

        return drift    # [window_num, 3]

    def _rcc(self, rmax: float = 0.8) -> ndarray:
        drift = self._getDriftMatrix()  # [window_num, window_num, 3]

        # number of drifts that is non-zero
        N = self.window_num * (self.window_num - 1) // 2
        # r, A
        r = np.zeros((N, 3))                    # [N, 3]
        A = np.zeros((N, self.window_num-1))    # [N, window_num-1]
        flag = 0
        for i in range(self.window_num-1):
            for j in range(i + 1, self.window_num):
                r[flag] = drift[i][j]
                A[flag, i:j] = 1
                flag += 1
        # error
        # get the error according to r and A
        error = A @ (np.linalg.pinv(A) @ r) - r     # [N, 3]
        error = np.linalg.norm(error, axis=1)       # [N]
        # col1 is norm error of each drift, col2 is the corresponding index
        error = np.array([[error[i], i] for i in range(N)])
        # we sort error in descending order where keep the index of the drift 
        # in order, i.e., still match to the err.
        error = error[np.flipud(np.argsort(error[:,0])),:]
        # select all the index where error is larger than rmax
        index = error[np.where(error[:,0] > rmax)[0]][:, 1].astype(int)
        for flag in index:
            # try to delete flag row in A
            # if the remaining matrix has a rank equal to nbinframe - 1, it 
            # means that deleting row flag will not cause the matrix A to become
            # singular then we can delete the corresponding row in A and r 
            temp = np.delete(A.copy(), flag, axis=0)
            if np.linalg.matrix_rank(temp) != (self.window_num - 1): continue
            # delete corresponding row in A and r
            A = np.delete(A, flag, axis=0)
            r = np.delete(r, flag, axis=0)
            # update index that we need to delete later since now we have less 
            # row and index larger than flag (current) should minus 1
            index[np.where(index > flag)[0]] -= 1
        # drift
        drift = np.linalg.pinv(A) @ r           # [window_num-1, 3]
        drift = np.cumsum(drift, axis=0)        # [window_num-1, 3]
        drift = np.insert(drift, 0, 0, axis=0)  # [window_num, 3]

        return drift    # [window_num, 3]

    def _getDriftMatrix(self) -> ndarray:
        # [window_num, window_num, 3]
        drift = np.zeros([self.window_num, self.window_num, 3])

        # load the drift vector from cache if exists and transform it to
        # drift matrix and return
        if os.path.exists(os.path.join(self.temp_save_fold, "r.csv")):
            r = np.loadtxt(
                os.path.join(self.temp_save_fold, "r.csv"), delimiter=','
            )
            flag = 0
            for i in range(self.window_num-1):
                for j in range(i+1, self.window_num):
                    drift[i][j] = r[flag]
                    flag += 1
            print(
                "Load drift matrix from `{}`. ".format(
                    os.path.join(self.temp_save_fold, "r.csv")
                ) + 
                "This drift matrix is temp result shared between MCC and RCC " + 
                "method to save drift calculation time. " +
                "Please ignore if you have run one of MCC or RCC method and " +
                "want to try another method. "
                "Delete `{}` if you want to re-calculate the drift ".format(
                    os.path.join(self.temp_save_fold, "r.csv")
                ) + 
                "for same dataset with new window size. " + 
                "Delete whole `{}` if you want ".format(self.temp_save_fold) + 
                "to re-calculate the drift for same dataset with new stride " + 
                "size. " + 
                "Delete whole `{}` or specify a ".format(self.temp_save_fold) + 
                "new path (recommend) if you want to re-calculate the drift " + 
                "for different dataset."
            )
            return drift

        for i in tqdm.tqdm(
            range(self.window_num), dynamic_ncols=True,
            desc=os.path.join(self.temp_save_fold, "r.csv")
        ):
            imagei = self._getWindow(i)
            for j in tqdm.tqdm(
                range(i, self.window_num), leave=False, 
                dynamic_ncols=True, smoothing=0.0,
            ):
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
                    else:
                        drift[i][j] = drift[i][j-1]
                    tqdm.tqdm.write(
                        "Optimal para not found for window ({},{})".format(i, j)
                    )
        # since the drift of window (i,i) must be zero, the center of 
        # drift[i][:] is drift[i][i] and need to be subtracted from drift[i][:] 
        for i in range(self.window_num):
            for j in range(i+1, self.window_num):
                drift[i][j] -= drift[i][i]
            drift[i][i] -= drift[i][i]

        # flat the drift matrix to drift vector to save for future use
        # number of drifts that is non-zero
        N = self.window_num * (self.window_num - 1) // 2
        r = np.zeros((N, 3))                    # [N, 3]
        flag = 0
        for i in range(self.window_num-1):
            for j in range(i+1, self.window_num):
                r[flag] = drift[i][j]
                flag += 1
        np.savetxt(
            os.path.join(self.temp_save_fold, "r.csv"), 
            r, delimiter=','
        )

        return drift

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
        bounds: tuple[tuple[float, ...], tuple[float, ...]] = None
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
        plt.title("Drift over frames (method: {})".format(self.method))
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
        plt.title("Drift in XY plane (method: {})".format(self.method))
        plt.xlabel('x (pixel)')
        plt.ylabel('y (pixel)')

        # show both plots simultaneously
        plt.show()
