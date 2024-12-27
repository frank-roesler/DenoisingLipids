import platform
import scipy
import numpy as np
import torch.cuda
from scipy.fft import ifft, fftshift
from src.utils.utils_infer import denoise_signal
import matplotlib.pyplot as plt
from scipy import io
import pathlib

myPlatform = platform.system()
if myPlatform == 'Darwin':
    pathlib.WindowsPath = pathlib.PosixPath
print(myPlatform)

NoiseFit     = True # Set to "True" if model was trained to fit the noise, "False" if trained to fit signal
DiffusionFit = False # Set to "True" if a 2d model was used to fit all b-values simultaneously

model_path = 'trained_models/L1/Unet/model.pth'
matPath    = 'data/__DATEN/mrsiData_Lip.mat'

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

pixel = 265
mat    = scipy.io.loadmat(matPath)
mrsSet = np.array(mat['mrsiData'][0][0][1][:,pixel+15: pixel+16])

y = ifft( np.conj( mrsSet ), axis=0)
y = fftshift(y, axes=0 )

# Denoise signal:
y_dn_cplx = denoise_signal(y, model, diffusion=DiffusionFit, noise_fit=NoiseFit, device=device)

fig, ax = plt.subplots(2,1,figsize=(14,6), constrained_layout=True)
ax[0].plot(np.real(y), linewidth=0.5)
ax[1].plot(np.real(y_dn_cplx), linewidth=0.5)
ax[0].set_xlim(len(y), 0)
ax[1].set_xlim(len(y), 0)
fig.savefig('L1_{}'.format(pixel), dpi=256)
# plt.show()