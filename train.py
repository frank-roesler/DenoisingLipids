import torch.optim as optim
import torch
import numpy as np
from utils_info import print_training_data, InfoScreen
from utils_simul import make_batch_diffusion, MMBG_basis, Metab_basis, Lip_basis, build_ppmAx
from utils_io import Checkpoint, load_model
from nets import DiffusionNet,UNet,DiffusionNet_compr
from parameter_values import *
from config_train import *
import matplotlib.pyplot as plt

metab_basis = Metab_basis(metab_path, kwargs_BS, metab_con, normalize_basis_sets=NormalizeBasisSets)
mmbg_basis  = MMBG_basis(mmbg_path, kwargs_MM, reduce_small_mm=ReduceSmallMMs) if includeMMBG else None
lip_basis   = Lip_basis(lip_path, kwargs_Lipd ) if includeLip else None

ppmAx, fAx, wCenter, fL = build_ppmAx(bw, noSmp)
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device: ', device)

# model = DiffusionNet(ks=(35,3), nc=64).to(device)
# model = UNet(n_classes=2,n_channels=2).to(device)
model = DiffusionNet_compr(ks1=(15,3), ks2=(16,3), nc=32).to(device)

loss_fn = torch.nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=True)

timer  = 0       # Number of steps before model can be saved again
epoch  = 0
losses = []
best_loss = torch.Tensor([1e-2]).to(device) # Threshold for saving the model
if LoadPretrainedModel:
    print_training_data(modelname)
    losses, epoch, current_loss, best_loss, batch_size = load_model(modelname, model, optimizer, device)
    
# checkpoint = torch.load(modelname)
# #model = UNet(n_classes=2,n_channels=2).to(device)
# model = DiffusionNet_compr().to(device)
# optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print('training...')
info_screen = InfoScreen(output_every=plot_loss_every, plot_spectra_during_train=plotSpectraDuringTraining)
checkpoint  = Checkpoint()
model.train()
while epoch <= epochs+1:
    if timer>2000:
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
        noisy_signal_batch = noisy_signal_batch.to(device)
        noise_batch        = noise_batch.to(device)
        lip_batch          = lip_batch.to(device)
        pred   = model(noisy_signal_batch)
        target = lip_batch + noise_batch
        loss  += loss_fn(pred, target)/len(bvals)
        info_screen.plot_spectra(n_bvals, noisy_signal_batch, target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
    optimizer.step()
    losses.append(loss.cpu().detach().data[0])
    info_screen.print_info(losses, optimizer, epoch, epochs, model, noise_batch.shape[0]*len(bvals))
    info_screen.plot_losses(epoch, losses)

    current_loss = np.mean(losses[-500:])
    timer = checkpoint.save(timer, current_loss, epoch, model, optimizer, losses, best_loss)
    timer += 1
    epoch += 1


plt.show()