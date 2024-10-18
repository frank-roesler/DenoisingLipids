from parameter_values import *
from config_train import *
import torch
import numpy as np
import os


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



class Checkpoint:
    def __init__(self):
        self.trainingData = {
            'batch_size': batch_size,
            'learning_rate': lr,
            'includeMMBG': includeMMBG,
            'includeLip': includeLip,
            'Monotonicity': Monotonicity,
            'NormalizeBasisSets': NormalizeBasisSets,
            'ReduceSmallMMs': ReduceSmallMMs,
            'metab_path': metab_path,
            'mmbg_path': mmbg_path,
            'lip_path': lip_path,
            'bandwidth': bw,
            'noSmp': noSmp,
            'kwargs_BS': kwargs_BS,
            'kwargs_MM': kwargs_MM,
            'kwargs_Lipd': kwargs_Lipd}

    def save(self, timer, current_loss, epoch, model, optimizer, losses, best_loss):
        if timer<100:
            return timer
        if current_loss > best_loss:
            return timer
        best_loss = current_loss
        dataLocal = {'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,
                'best_loss': best_loss}
        torch.save(self.trainingData.update(dataLocal), os.path.join(modeldir, model.name))
        print('new best loss: ', "{:.3e}".format(best_loss))
        return 0