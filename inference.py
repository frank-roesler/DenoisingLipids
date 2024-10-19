import scipy
import numpy as np
import torch.cuda
from scipy.fft import fft, ifft, fftshift
from nets import DiffusionNet, UNet
from utils_infer import denoise_signal
import matplotlib.pyplot as plt
from scipy import io

NoiseFit     = True # Set to "True" if model was trained to fit the noise, "False" if trained to fit signal
DiffusionFit = True # Set to "True" if a 2d model was used to fit all b-values simultaneously

model_path = 'trained_models/DiffusionNet_compr_15x3_16x3_32'
matPath    = 'result.mat'

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
checkpoint = torch.load(model_path, map_location=device)

model = checkpoint['model']
model.eval()
print(model)

mat    = scipy.io.loadmat(matPath)
mrsSet = np.array(mat['metabData'])
y = fftshift(ifft( np.conj( mrsSet ) ), 1 )

# Denoise signal:
y_dn_cplx = denoise_signal(y, model, diffusion=DiffusionFit, noise_fit=NoiseFit, device=device)


plt.plot(np.real(y_dn_cplx[:,1500:2500].T), linewidth=0.5)
plt.show()