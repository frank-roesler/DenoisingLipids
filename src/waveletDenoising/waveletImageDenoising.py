from skimage.restoration import denoise_wavelet
from pywt import wavedec, waverec, coeffs_to_array, dwt_max_level, wavelist, cwt
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import platform
import scipy
from scipy.fft import ifft, fftshift
from PIL import Image


imgPath = "/Users/frankrosler/Desktop/PhD/Python/denoising_Sethian/IMG_1638_large.JPG"

wavelet = "db2"
sigma = 0.4
waveletLevels = 7


def isGrayScale(img: Image) -> bool:
    img = img.convert("RGB")
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            if r != g != b:
                return False
    return True


def convertToGrayScale(imgPath: str) -> Image:
    img = Image.open(imgPath)
    if not isGrayScale(img):
        img = img.convert("L")
    return np.array(img) / 255.0


imgData = convertToGrayScale(imgPath)
imgData += sigma * np.random.randn(*imgData.shape)

print("Wavelet levels:", dwt_max_level(len(imgData), wavelet))
print("-" * 100)

imgDenoised = denoise_wavelet(imgData, wavelet=wavelet, sigma=0.3, mode="soft", wavelet_levels=waveletLevels)

fig, ax = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
im0 = ax[0].imshow(imgData)
im1 = ax[1].imshow(imgDenoised)
ax[0].axis("off")
ax[1].axis("off")
im0.set_clim(0, 1)
im1.set_clim(0, 1)
plt.show()
