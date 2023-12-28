import numpy as np
import cupy as cp
import scipy.optimize
import scipy.interpolate
import os
import tifffile
import matplotlib.pyplot as plt
import tqdm

__all__ = ["DriftCorrector"]


class DriftCorrector:
    def __init__(self, total, window, step, crop):
        self.total = total
        self.window = window
        self.step = step
        self.crop = crop
        self.window_num = (self.total-self.window)//self.step+1

        self.index_src = self.window/2 + np.arange(self.window_num)*self.step
        self.index_dst = np.arange(total) + 1
        self.drift_src = None
        self.drift_dst = None

        # preload all images to memory
        self.image = [
            tifffile.imread("D:/drift/{}".format(i)) 
            for i in os.listdir('D:/drift/')
        ]

    def fit(self):
        self._dcc()
        self._interpolation()
        self._plot()

    def _dcc(self) -> None:
        drift = []  # [window_num, 3]

        image0 = self._getWindow(0)
        for j in tqdm.tqdm(range(1, self.window_num)):
            imagej = self._getWindow(j)
            # calculate the cross correlation
            corr = self.crossCorrelation3D(image0, imagej)
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
            drift_0j  = DriftCorrector.gaussianFit(corr)
            drift_0j -= ((np.array(self.crop)-1)/2)
            # add drift to data structure
            drift.append(drift_0j)

        drift = np.array(drift)
        drift = np.insert(drift, 0, 0, axis=0)
        self.drift_src = drift

    def _mcc(self) -> None:
        # [window_num, window_num-1, 3], will sum to [window_num, 3] when return
        drift = [[] for _ in range(self.window_num)]

        for i in tqdm.tqdm(range(self.window_num)):
            imagei = self._getWindow(i)
            for j in tqdm.tqdm(range(i+1, self.window_num), leave=False):
                print(i, j)
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
                drift_ij  = DriftCorrector.gaussianFit(corr)
                drift_ij -= ((np.array(self.crop)-1)/2)
                # add drift to data structure
                drift[i].append( drift_ij)  # drift(i, j)
                drift[j].append(-drift_ij)  # drift(j, i)

        drift = np.array(drift).sum(axis=1)
        drift = np.insert(drift, 0, 0, axis=0)
        self.drift_src = drift

    def _getWindow(self, index):
        image = self.image[index]
        for i in range(1, self.window // self.step):
            image += self.image[index+i]
        return image

    @staticmethod
    def crossCorrelation3D(image1, image2):
        fft_image1 = cp.fft.fftn(cp.asarray(image1))
        fft_image2 = cp.fft.fftn(cp.asarray(image2))
        corr = cp.fft.ifftn(cp.multiply(fft_image1, cp.conj(fft_image2)))
        corr = cp.fft.fftshift(cp.real(corr))
        return cp.asnumpy(corr)

    @staticmethod
    def gaussianFit(corr):
        xdata = np.vstack(np.indices(corr.shape).reshape(3, -1))
        ydata = corr.ravel()
        p0 = (
            (corr.shape[0]-1)/2, (corr.shape[1]-1)/2, (corr.shape[2]-1)/2, 
            1, 1, 1, 1
        )
        popt, _ = scipy.optimize.curve_fit(
            DriftCorrector.gaussian3D, xdata, ydata, p0=p0, maxfev=3000
        )
        return popt[0:3]

    @staticmethod
    def gaussian3D(xyz, x0, y0, z0, sigma_x, sigma_y, sigma_z, amp):
        return amp * np.exp(-(
            (xyz[0] - x0) ** 2 / (2 * sigma_x ** 2) +
            (xyz[1] - y0) ** 2 / (2 * sigma_y ** 2) +
            (xyz[2] - z0) ** 2 / (2 * sigma_z ** 2)
        ))

    def _interpolation(self):
        interp_func_z = scipy.interpolate.interp1d(
            self.index_src, self.drift_src[:, 0], kind='cubic', 
            fill_value=(self.drift_src[0, 0], self.drift_src[-1, 0]), 
            bounds_error=False
        )
        interp_func_y = scipy.interpolate.interp1d(
            self.index_src, self.drift_src[:, 1], kind='cubic', 
            fill_value=(self.drift_src[0, 1], self.drift_src[-1, 1]), 
            bounds_error=False
        )
        interp_func_x = scipy.interpolate.interp1d(
            self.index_src, self.drift_src[:, 2], kind='cubic', 
            fill_value=(self.drift_src[0, 2], self.drift_src[-1, 2]), 
            bounds_error=False
        )

        drift_dst = np.zeros([self.total, 3])
        drift_dst[:, 0] = interp_func_z(self.index_dst)
        drift_dst[:, 1] = interp_func_y(self.index_dst)
        drift_dst[:, 2] = interp_func_x(self.index_dst)

        self.drift_dst = drift_dst

    def _plot(self):
        plt.figure()
        plt.plot(self.index_dst, self.drift_dst[:, 0], label='Z')
        plt.plot(self.index_dst, self.drift_dst[:, 1], label='Y')
        plt.plot(self.index_dst, self.drift_dst[:, 2], label='X')
        plt.legend()
        plt.title("total: {}; window: {}; step: {}; crop: {}".format(
            self.total, self.window, self.step, self.crop
        ))
        plt.xlabel('window index (frame)')
        plt.ylabel('drift (pixel)')
        plt.show()
