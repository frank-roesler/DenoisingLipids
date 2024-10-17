import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy
from scipy.fft import ifft, fftshift, fft, ifftshift
from scipy.stats import shapiro
from glob import glob
import os
from utils_simul import build_ppmAx



def moving_average(window_size, signal, mode='same'):
    """computes moving average of a signal with window of width window_size"""
    window = np.ones(window_size) / window_size
    signal_smoothed = np.convolve(signal, window, mode=mode)
    return signal_smoothed


def plot_losses(losses, mode='log'):
    t = range(len(losses))
    window_size = 500
    losses_smooth = moving_average(window_size, losses, mode='valid')
    tt = range(window_size // 2, len(losses_smooth) + window_size // 2)
    plt.cla()
    if mode == 'log':
        plt.semilogy(t, losses, linewidth=0.5)
        plt.semilogy(tt, losses_smooth, linewidth=1.5)
        # plt.ylim([1e-1,1e-0])
    else:
        plt.plot(t, losses, linewidth=0.5)
        plt.plot(tt, losses_smooth, linewidth=1.5)
        # plt.ylim([-3.9, 0])
    plt.title('Loss')
    plt.show(block=False)
    plt.pause(0.01)


def print_info(losses, optimizer, epoch, epochs, model):
    n_params = sum(p.numel() for p in model.parameters())
    print('=' * 50)
    print('Training. Epoch: ', str(epoch) + '/' + str(epochs), ', ', 'Loss: ', "{:.3e}".format(np.mean(losses[-500:])))
    print('-' * 50)
    print('Model: ', model.name)
    print('Number of model parameters: ', n_params)
    for param_group in optimizer.param_groups:
        print('Learning rate:', param_group['lr'])


def print_training_data(path, plot_loss=False):
    data = torch.load(path, map_location=torch.device("cpu"))
    print('='*100)
    print('TRAINED WITH:')
    modelname = path.split('/')[-1]
    print('model: ', modelname)
    for key, value in data.items():
        if key not in ('model_state_dict','optimizer_state_dict'):
            if key=='losses':
                print('final loss: ', "{:.2e}".format(np.mean(value[-400:])))
            else:
                print(key, ': ', value)
    if plot_loss:
        plot_losses(data['losses'])
        plt.show()
    print('=' * 100)


def print_empirical_noise_lvl(signal):
    maxValues = np.array(np.max(np.abs(signal), 1))
    for i in range(signal.shape[0]):
        noise = signal[i,2200:3400]/maxValues[i]
        print('Empirical Noise Lvl: ', np.real(noise).std())


def load_model(path, model, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        print('State dict did not fit optimizer. Not loaded.')
    epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    best_loss = checkpoint['best_loss']
    batch_size = checkpoint['batch_size']
    current_loss = np.mean(losses[-100:])
    return losses, epoch, current_loss, best_loss, batch_size



