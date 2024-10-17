import torch.optim as optim
import torch
import numpy as np
from utils_info import plot_losses, print_info, print_training_data, load_model
from utils_simul import make_batch_diffusion, MMBG_basis, Metab_basis, Lip_basis, build_ppmAx
from nets import DiffusionNet,UNet,DiffusionNet_compr
from parameter_values import *
import matplotlib.pyplot as plt
from time import time

includeMMBG         = False
includeLip          = True
LoadPretrainedModel = False
Monotonicity        = False
NormalizeBasisSets  = False  # normalizes all basis sets so that highest peak is 1. LEAVE THIS AT FALSE!!!
ReduceSmallMMs      = False  # Removes MMs with small amplitude to speed up training

metab_basis = Metab_basis(metab_path, kwargs_BS, metab_con, normalize_basis_sets=NormalizeBasisSets)
mmbg_basis  = MMBG_basis(mmbg_path, kwargs_MM, reduce_small_mm=ReduceSmallMMs) if includeMMBG else None
lip_basis   = Lip_basis(lip_path, kwargs_Lipd ) if includeLip else None

# check metab_basis
# fig, ax = plt.subplots(1,1, figsize=(10,6))
# ax.plot(metab_basis.naked_patterns, linewidth=0.5)
# plt.show(block=False)
# plt.pause(1)

batchSz = 4
imgRes  = 16   # should be 1024=32x32, 406, 2

noisy_signal_batch, noise_batch, lip_batch = make_batch_diffusion( batchSz, imgRes, metab_basis, mmbg_basis, lip_basis,
                                                                   restrict_range=None, #(1500,2500), 
                                                                   #restrict_range=(0,404), 
                                                                   include_mmbg = includeMMBG,
                                                                   include_lip  = includeLip,
                                                                   normalization='max_1', monotone_diffusion=Monotonicity,
                                                                   **kwargs_BS)

# plot result
ppm, fAx, wCenter, fL = build_ppmAx(bw, noSmp)



n_bvals = noisy_signal_batch.shape[-1]
M_mean = 0
cmap = plt.get_cmap('winter',n_bvals)
for B in range(batchSz):
    fig, ax = plt.subplots(2,1, figsize=(10,6))
    Max = np.max(torch.sqrt(noisy_signal_batch[B,0,:,0]**2 + noisy_signal_batch[B,1,:,0]**2).numpy())
    M_mean += Max/batchSz
    for i in range(n_bvals):
        ax[0].plot(ppm, noisy_signal_batch[B,0,:,i], linewidth=0.5, color=cmap(i))
        ax[0].plot(ppm, noisy_signal_batch[B,0,:,i]-noise_batch[B,0,:,i], '--', linewidth=0.5, color=cmap(i))
        ax[0].plot(ppm, noisy_signal_batch[B,0,:,i]-noise_batch[B,0,:,i]-lip_batch[B,0,:,i], '-.', linewidth=0.5, color=cmap(i))
        ax[1].plot(ppm, noisy_signal_batch[B,1,:,i], linewidth=0.5, color=cmap(i))
        ax[1].plot(ppm, noisy_signal_batch[B,1,:,i]-noise_batch[B,1,:,i]-lip_batch[B,1,:,i], '-.',linewidth=0.5, color=cmap(i))
    ax[0].set_xlim(max(ppm), min(ppm) )
    ax[1].set_xlim(max(ppm), min(ppm) )
    ax[0].set_xlim(4.8, 1.0)
    ax[1].set_xlim(4.8, 1.0)
    ax[0].set_ylim(-1,1)
    ax[1].set_ylim(-1,1)
plt.show(block=False)
plt.pause(1)

print("")