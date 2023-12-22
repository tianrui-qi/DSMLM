import numpy as np
import cupy as cp
import scipy.optimize
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
        self.window_index = self.window/2 + np.arange(self.window_num)*self.step
        self.drift = None

        # preload all images to memory
        self.image = [
            tifffile.imread("D:/drift/{}".format(i)) 
            for i in os.listdir('D:/drift/')
        ]

    def dcc(self):
        self.drift = []  # [window_num, 3]

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
            drift_0j = self.gaussianFit(corr) - ((np.array(self.crop)-1)/2)
            # add drift to data structure
            self.drift.append(drift_0j)

        self.drift = np.array(self.drift)
        self.drift = np.insert(self.drift, 0, 0, axis=0)
        return self.drift

    def mcc(self):
        # [window_num, window_num-1, 3], will sum to [window_num, 3] when return
        self.drift = [[] for _ in range(self.window_num-1)]

        for i in tqdm.tqdm(range(self.window_num)):
            imagei = self._getWindow(i)
            for j in tqdm.tqdm(range(i+1, self.window_num), leave=False):
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
                drift_ij = self.gaussianFit(corr) - ((np.array(self.crop)-1)/2)
                # add drift to data structure
                self.drift[i].append( drift_ij)  # drift(i, j)
                self.drift[j].append(-drift_ij)  # drift(j, i)

        self.drift = np.array(self.drift).sum(axis=1)
        self.drift = np.insert(self.drift, 0, 0, axis=0)
        return self.drift

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
        def gaussian3D(xyz, x0, y0, z0, sigma_x, sigma_y, sigma_z, amp):
            return amp * np.exp(-(
                (xyz[0] - x0) ** 2 / (2 * sigma_x ** 2) +
                (xyz[1] - y0) ** 2 / (2 * sigma_y ** 2) +
                (xyz[2] - z0) ** 2 / (2 * sigma_z ** 2)
            ))
        xdata = np.vstack(np.indices(corr.shape).reshape(3, -1))
        ydata = corr.ravel()
        p0 = (
            (corr.shape[0]-1)/2, (corr.shape[1]-1)/2, (corr.shape[2]-1)/2, 
            1, 1, 1, 1
        )
        popt, _ = scipy.optimize.curve_fit(gaussian3D, xdata, ydata, p0=p0)
        return popt[0:3]

    def plotDrift(self):
        if self.drift is None:
            raise Exception("drift is None, please run dcc() or mcc() first")

        plt.figure()
        plt.plot(self.window_index, self.drift[:, 0], label='Z')
        plt.plot(self.window_index, self.drift[:, 1], label='Y')
        plt.plot(self.window_index, self.drift[:, 2], label='X')
        plt.legend()
        plt.title("total: {}; window: {}; step: {}; crop: {}".format(
            self.total, self.window, self.step, self.crop
        ))
        plt.xlabel('window index (frame)')
        plt.ylabel('drift (pixel)')
        plt.show()
