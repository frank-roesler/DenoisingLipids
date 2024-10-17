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
includeLip          = False
LoadPretrainedModel = False
Monotonicity        = False
NormalizeBasisSets  = False  # normalizes all basis sets so that highest peak is 1. LEAVE THIS AT FALSE!!!
ReduceSmallMMs      = False  # Removes MMs with small amplitude to speed up training

trainLs = True  # train the network for lipid suppresion (otherwise it's just denoising)

modelname = '03_SLOW/model/DiffusionNet_compr_33x7_34x7_32' # load this model
modeldir  = '03_SLOW/model/' # save model as

metab_basis = Metab_basis(metab_path, kwargs_BS, metab_con, normalize_basis_sets=NormalizeBasisSets)
mmbg_basis  = MMBG_basis(mmbg_path, kwargs_MM, reduce_small_mm=ReduceSmallMMs) if includeMMBG else None
lip_basis   = Lip_basis(lip_path, kwargs_Lipd ) if includeLip else None
bvals = range(32,32+1)
#bvals = range(7,7+1)

ppmAx, fAx, wCenter, fL = build_ppmAx(bw, noSmp)
device = torch.device('mps' if torch.mps.is_available() else 'cpu')
epochs     = 100000
lr         = 6e-5
batch_size = 32    # will be multiplied by n_bvals
print('device: ', device)

model = DiffusionNet(ks=(35,3), nc=64).to(device)
# model = UNet(n_classes=2,n_channels=2).to(device)
# model = DiffusionNet_compr().to(device)
loss_fn = torch.nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=True)

timer  = 0       # Number of steps before model can be saved again
epoch  = 0
losses = []
best_loss = 1e-2 # Threshold for saving the model
if LoadPretrainedModel:
    print_training_data(modelname)
    losses, epoch, current_loss, best_loss, batch_size = load_model(modelname, model, optimizer, device)
    
# checkpoint = torch.load(modelname)
# #model = UNet(n_classes=2,n_channels=2).to(device)
# model = DiffusionNet_compr().to(device)
# optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

t0 = time()
print('training...')
model.train()
# fig, ax = plt.subplots(2,1,figsize=(14,6), constrained_layout=True)
while epoch <= epochs+1:
    if timer>2000:
        # If best_loss hasn't been beaten in the last 1000 steps, increase batch size.
        if batch_size==128:
            break
        batch_size *= 2
        timer = 100
    loss = torch.zeros(1, device=device)
    for n_bvals in bvals:
        noisy_signal_batch, noise_batch, lip_batch = make_batch_diffusion( batch_size, n_bvals, metab_basis, mmbg_basis, lip_basis,
                                                                           restrict_range=None, #(1500,2500), 
                                                                           #restrict_range=(0,404), 
                                                                           include_mmbg = includeMMBG,
                                                                           include_lip  = includeLip,
                                                                           normalization='max_1', monotone_diffusion=Monotonicity,
                                                                           **kwargs_BS)
        
        noise_batch        = noise_batch.to(device)

        if trainLs:
            noisy_signal_batch = noisy_signal_batch - lip_batch

        noisy_signal_batch = noisy_signal_batch.to(device)
        pred = model(noisy_signal_batch)
        loss += loss_fn(pred, noise_batch)/len(bvals)

        # cmap = plt.get_cmap('winter', n_bvals)
        # S = noisy_signal_batch[0][0].detach().cpu()
        # N = noisy_signal_batch[0][0].detach().cpu() - noise_batch[0][0].detach().cpu()
        # ax[0].cla()
        # ax[1].cla()
        # low = 1500
        # high = 2500
        # for b in range(n_bvals):
        #     ax[0].plot(ppmAx[low:high], S[low:high, b], linewidth=0.5, color=cmap(b))
        #     ax[1].plot(ppmAx[low:high], N[low:high, b], linewidth=0.5, color=cmap(b))
        # plt.show(block=False)
        # plt.pause(1)
        # ax[0].set_xlim(ppmRange[1] + 0.5, ppmRange[0] - 0.5)  # ppm axis
        # ax[1].set_xlim(ppmRange[1] + 0.5, ppmRange[0] - 0.5)  # ppm axis
        # fig.savefig('without{}{}'.format(epoch,n_bvals), dpi=256)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
    optimizer.step()
    losses.append(float(loss))
    del loss
    if epoch%1000==0 and epoch>0:
        print_info(losses,optimizer,epoch,epochs,model)
        print('batch size: ', noise_batch.shape[0]*len(bvals))
        print('Time: ', time() - t0)
        print('-' * 50)
        t0 = time()
        plot_losses(losses, mode='log')

    current_loss = np.mean(losses[-500:])
    if current_loss<best_loss and timer>100:
        timer = 0
        best_loss = current_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
            'best_loss': best_loss,
            'batch_size': batch_size,
            'learning_rate': lr,
            'includeMMBG': includeMMBG,
            'Monotonicity': Monotonicity,
            'NormalizeBasisSets': NormalizeBasisSets,
            'ReduceSmallMMs': ReduceSmallMMs,
            'metab_path': metab_path,
            'mmbg_path': mmbg_path,
            'bandwidth': bw,
            'kwargs_BS': kwargs_BS,
            'kwargs_MM': kwargs_MM
        }, modeldir+model.name)
        print('new best loss: ', "{:.3e}".format(best_loss))
    timer += 1
    epoch += 1
plt.show()