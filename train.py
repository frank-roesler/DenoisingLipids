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

epochs     = 100000
lr         = 6e-5
batch_size = 32    # will be multiplied by n_bvals

trainLs = True  # train the network for lipid suppresion (otherwise it's just denoising)

modelname = '03_SLOW/model/DiffusionNet_compr_33x7_34x7_32' # load this model
modeldir  = '03_SLOW/model/' # save model as

metab_basis = Metab_basis(metab_path, kwargs_BS, metab_con, normalize_basis_sets=NormalizeBasisSets)
mmbg_basis  = MMBG_basis(mmbg_path, kwargs_MM, reduce_small_mm=ReduceSmallMMs) if includeMMBG else None
lip_basis   = Lip_basis(lip_path, kwargs_Lipd ) if includeLip else None
bvals = range(32,32+1)
#bvals = range(7,7+1)

ppmAx, fAx, wCenter, fL = build_ppmAx(bw, noSmp)
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device: ', device)

# model = DiffusionNet(ks=(35,3), nc=64).to(device)
# model = UNet(n_classes=2,n_channels=2).to(device)
model = DiffusionNet_compr(ks1=(15,32), ks2=(16,32), nc=32).to(device)
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

t0 = time()
print('training...')
model.train()
# fig, ax = plt.subplots(2,1,figsize=(14,6), constrained_layout=True)
while epoch <= epochs+1:

    print('start while ', time() - t0)
    t0 = time()

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
        print('end make_batch ', time()-t0)
        t0 = time()

        noise_batch        = noise_batch.to(device)
        noisy_signal_batch = noisy_signal_batch.to(device)
        lip_batch          = lip_batch.to(device)

        pred   = model(noisy_signal_batch)
        target = lip_batch + noise_batch
        loss += loss_fn(pred, target)/len(bvals)

        # cmap = plt.get_cmap('winter', n_bvals)
        # S = noisy_signal_batch[0][0].detach().cpu()
        # N = target[0][0].detach().cpu()
        # ax[0].cla()
        # ax[1].cla()
        # low = 1500
        # high = 2500
        # for b in range(n_bvals):
        #     ax[0].plot(S[:, b], linewidth=0.5, color=cmap(b))
        #     ax[1].plot(N[:, b], linewidth=0.5, color=cmap(b))
        # ax[0].set_xlim(len(S), 0)  # ppm axis
        # ax[1].set_xlim(len(N), 0)  # ppm axis
        # plt.show(block=False)
        # plt.pause(1)
        # fig.savefig('lipids{}{}'.format(epoch,n_bvals), dpi=256)

    print('end compute loss ', time()-t0)
    t0 = time()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    print('end backward ', time()-t0)
    t0 = time()

    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
    optimizer.step()

    print('end optimizer ', time()-t0)
    t0 = time()

    loss_fl = loss

    print('end float(loss) ', time()-t0)
    t0 = time()

    losses.append(loss_fl)

    print('end losses.append() ', time()-t0)
    t0 = time()

    del loss

    print('end del loss ', time()-t0)
    t0 = time()

    if epoch%100==0 and epoch>0:
        print_info(losses,optimizer,epoch,epochs,model)
        print('batch size: ', noise_batch.shape[0]*len(bvals))
        print('Time: ', time() - t0)
        print('-' * 50)
        t0 = time()
        plot_losses(losses, mode='log')

    current_loss = torch.mean(torch.stack(losses[-500:]))

    print('end current_loss ', time()-t0)
    t0 = time()

    optimal = current_loss < best_loss

    print('end optimal ', time()-t0)
    print(optimal)
    t0 = time()

    # if timer>100:
    #     if optimal:
    #         timer = 0
    #         best_loss = current_loss
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'losses': losses,
    #             'best_loss': best_loss,
    #             'batch_size': batch_size,
    #             'learning_rate': lr,
    #             'includeMMBG': includeMMBG,
    #             'Monotonicity': Monotonicity,
    #             'NormalizeBasisSets': NormalizeBasisSets,
    #             'ReduceSmallMMs': ReduceSmallMMs,
    #             'metab_path': metab_path,
    #             'mmbg_path': mmbg_path,
    #             'bandwidth': bw,
    #             'kwargs_BS': kwargs_BS,
    #             'kwargs_MM': kwargs_MM
    #         }, modeldir+model.name)
    #     print('new best loss: ', "{:.3e}".format(best_loss))

    timer += 1
    epoch += 1

    print('end for step ', time()-t0)
    t0 = time()
    print('-'*100)


plt.show()