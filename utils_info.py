import numpy as np
import matplotlib.pyplot as plt
import torch
from time import time



def moving_average(window_size, signal, mode='same'):
    """computes moving average of a signal with window of width window_size"""
    window = np.ones(window_size) / window_size
    signal_smoothed = np.convolve(signal, window, mode=mode)
    return signal_smoothed


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
        info_screen = InfoScreen(output_every=1)
        info_screen.plot_losses(data['losses'])
        plt.show()
    print('=' * 100)


def print_empirical_noise_lvl(signal):
    maxValues = np.array(np.max(np.abs(signal), 1))
    for i in range(signal.shape[0]):
        noise = signal[i,2200:3400]/maxValues[i]
        print('Empirical Noise Lvl: ', np.real(noise).std())


class InfoScreen:
    def __init__(self, output_every=1):
        self.t0 = time()
        self.t1 = time()
        self.output_every = output_every
        self.init_plots()

    def init_plots(self):
        self.fig, self.ax = plt.subplots(1,1, figsize = (11,5), constrained_layout=True)
        self.p1 = self.ax.semilogy([0],[1], label='training loss', linewidth=0.5)[0]
        self.s1 = self.ax.semilogy([0],[1], label='running mean')[0]
        self.ax.legend()
        self.ax.set_title('Loss')

    def print_info(self, losses, optimizer, epoch, epochs, model, batch_size):
        if epoch%self.output_every!=0:
            return
        self.t1 = time() - self.t0
        self.t0 = time()
        n_params = sum(p.numel() for p in model.parameters())
        print('=' * 50)
        print('Training. Epoch: ', str(epoch) + '/' + str(epochs), ', ', 'Loss: ', "{:.3e}".format(np.mean(losses[-500:])))
        print('.........', 'Model: ', model.name)
        print('.........', 'Number of model parameters: ', n_params)
        for param_group in optimizer.param_groups:
            print('.........', 'Learning rate:', param_group['lr'])
        print('batch size: ', batch_size)
        print('.........', f'Time: {self.t1:.1f}')

    def plot_losses(self, epoch, train_losses, window=100):
        """plots loss and accuracy curves during training, along with their running means."""
        if epoch>0 and epoch%self.output_every==0:
            self.ax.set_xlim(1,epoch+1)
            self.ax.set_ylim((0.9*np.min(train_losses), 1.1*np.max(train_losses)))
            self.p1.set_xdata(range(1, epoch + 1))
            self.p1.set_ydata(train_losses)
            if epoch>2*window+1:
                self.s1.set_xdata(range(window, epoch + 1 - window))
                self.s1.set_ydata(np.convolve(train_losses, np.ones(2 * window) / (2 * window), mode='valid'))
            self.fig.canvas.draw()
            plt.show(block=False)
            plt.pause(0.001)



