from skimage.restoration import denoise_wavelet
from pywt import dwt_max_level
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import platform
import scipy
from scipy.fft import ifft, fftshift

myPlatform = platform.system()
if myPlatform == "Darwin":
    pathlib.WindowsPath = pathlib.PosixPath
print(myPlatform)

# matPath = "data/__DATEN/mrsiData_Lip.mat"
matPath = "/Users/frankrosler/Desktop/PhD/Python/denoising_voigt/Daten/STEAM.nosync/new/2022_04_06_02_InVivo_AnomDiff/dSTEAM_D050/result.mat"
wavelet = "db2"
sigma = 1.04e-8
waveletLevels = None


def compute_spectrum(timeDomainSignal):
    y = ifft(np.conj(timeDomainSignal), axis=0)
    y = fftshift(y, axes=0)
    return y


mat = scipy.io.loadmat(matPath)
mrsSet = np.array(mat["metabData"])[2]
# mrsSet = np.array(mat['metab'])
# whichPixel = 256
# mrsSet = np.array(mat["mrsiData"][0][0][1][:, whichPixel + 15 : whichPixel + 16])
y = compute_spectrum(mrsSet)

print(mrsSet.shape)
print("NoiseLvl time domain: ", np.std(mrsSet[-500:]))
print("NoiseLvl frequency domain: ", np.std(y[-1000:-500]))
print("Wavelet levels:", dwt_max_level(len(mrsSet), wavelet))
print("-" * 100)


# # ------------------------------------------------
# # Denoise Time domain signal and compute spectrum:
# # ------------------------------------------------
# mrsSetDenoisedReal = denoise_wavelet(
#     np.real(mrsSet), wavelet=wavelet, sigma=sigma, mode="soft", wavelet_levels=waveletLevels
# )
# mrsSetDenoisedImag = denoise_wavelet(
#     np.imag(mrsSet), wavelet=wavelet, sigma=sigma, mode="soft", wavelet_levels=waveletLevels
# )
# yDenoised = compute_spectrum(mrsSetDenoisedReal + 1j * mrsSetDenoisedImag)

# fig, ax = plt.subplots(4, 1, figsize=(14, 8), constrained_layout=True)
# ax[0].plot(np.real(mrsSet), linewidth=0.5)
# ax[1].plot(np.real(y - yDenoised), linewidth=0.5)
# ax[2].plot(np.real(y), linewidth=0.5)
# ax[3].plot(np.real(yDenoised), linewidth=0.5)

# ax[2].set_xlim(len(y), 0)
# ax[3].set_xlim(len(yDenoised), 0)
# ax[1].set_ylim(np.min(y), np.max(y))
# plt.show()


# ------------------------------------------------
# Denoise spectrum directly:
# ------------------------------------------------
yDenoisedReal = denoise_wavelet(np.real(y), wavelet=wavelet, sigma=sigma, mode="soft", wavelet_levels=waveletLevels)
yDenoisedImag = denoise_wavelet(np.imag(y), wavelet=wavelet, sigma=sigma, mode="soft", wavelet_levels=waveletLevels)
yDenoised = yDenoisedReal + 1j * yDenoisedImag

fig, ax = plt.subplots(3, 1, figsize=(14, 8), constrained_layout=True)
# ax[0].plot(np.real(mrsSet), linewidth=0.5)
ax[2].plot(np.real(y - yDenoised), linewidth=0.5)
ax[0].plot(np.real(y), linewidth=0.5)
ax[1].plot(np.real(yDenoised), linewidth=0.5)
ax[0].set_xlim(2300, 1500)
ax[1].set_xlim(2300, 1500)
ax[2].set_xlim(2300, 1500)
ax[0].title.set_text("Noisy Data")
ax[1].title.set_text("Wavelet denoised Data")
ax[2].title.set_text("Difference")
fig.savefig(f"waveletDenoising.png", dpi=300, bbox_inches="tight", pad_inches=0)
plt.show()
