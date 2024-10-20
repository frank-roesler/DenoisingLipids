import torch
import numpy as np


def complex_to_tensor(signal, normalize=True, diffusion=False):
    if normalize:
        if diffusion:
            maxValues = np.max(np.abs(signal), axis=0)
        else:
            maxValues = np.array(np.max(np.abs(signal), 1))[np.newaxis]
        signal = signal / maxValues.T
    if diffusion:
        y_tensor = torch.from_numpy(signal.squeeze()).cfloat()
        y_tensor = torch.stack([torch.real(y_tensor), torch.imag(y_tensor)]).unsqueeze(0)
        # y_tensor = np.transpose(y_tensor, (0,1,3,2)) # (1, 2, 4096, 6)
    else:
        y_tensor = torch.from_numpy(signal.squeeze()).cfloat()
        y_tensor = torch.stack([torch.real(y_tensor), torch.imag(y_tensor)])
        y_tensor = np.transpose(y_tensor, (1, 0, 2)) # (6, 2, 4096)
    if normalize:
        return y_tensor, maxValues
    else:
        return y_tensor

def denoise_signal(signal, model, diffusion=False, device=torch.device("cpu"), noise_fit=False):
    """applies trained model to noisy spectra"""
    with torch.no_grad():
        y_tensor, maxValues = complex_to_tensor(signal, diffusion=diffusion)

        if noise_fit:
            noise = model(y_tensor.to(device)).cpu()
            y_dn = y_tensor-noise
        else:
            y_dn = model(y_tensor.to(device))
        y_dn = y_dn.detach().cpu().numpy()
        if diffusion:
            y_dn = y_dn.squeeze()
            y_dn = np.transpose(y_dn, (2,0,1))
        y_dn_cplx = (y_dn[:, 0, :] + y_dn[:, 1, :]* 1j).squeeze()
        y_dn_cplx = y_dn_cplx.T * maxValues
    return y_dn_cplx