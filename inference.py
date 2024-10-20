import platform

import scipy
import numpy as np
import torch.cuda
from scipy.fft import fft, ifft, fftshift
from nets import DiffusionNet, UNet
from utils_infer import denoise_signal
import matplotlib.pyplot as plt
from scipy import io
from pathlib import Path
import pathlib
myPlatform = platform.system()
if myPlatform == 'Darwin':
    pathlib.WindowsPath = pathlib.PosixPath
print(myPlatform)

NoiseFit     = True # Set to "True" if model was trained to fit the noise, "False" if trained to fit signal
DiffusionFit = True # Set to "True" if a 2d model was used to fit all b-values simultaneously

model_path = 'trained_models/DiffusionNet_compr_15x3_16x3_32/model.pth'
matPath    = 'data/__DATEN/mrsiData_Lip.mat'

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

mat    = scipy.io.loadmat(matPath)
mrsSet = np.array(mat['mrsiData'][0][0][1][:,492:498])

print(mrsSet)

y = ifft( np.conj( mrsSet ), axis=0)
y = fftshift(y, axes=0 )
plt.plot(np.real(y), linewidth=0.5)
plt.show()
# Denoise signal:
y_dn_cplx = denoise_signal(y, model, diffusion=DiffusionFit, noise_fit=NoiseFit, device=device)
#
#
plt.plot(np.real(y_dn_cplx), linewidth=0.5)
plt.show()