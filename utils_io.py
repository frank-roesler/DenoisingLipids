from parameter_values import *
from config_train import *
import torch
import numpy as np
import os


def load_model(path, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    model = checkpoint['model']
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        print('State dict did not fit optimizer. Not loaded.')
    epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    best_loss = checkpoint['best_loss']
    batch_size = checkpoint['batch_size']
    current_loss = np.mean(losses[-100:])
    return model, losses, epoch, current_loss, best_loss, batch_size



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
            'kwargs_Lipid': kwargs_Lipid}

    def save(self, timer, current_loss, epoch, model, optimizer, losses, best_loss):
        if timer<100:
            return timer
        if current_loss > best_loss:
            return timer
        best_loss = current_loss
        dataLocal = {'epoch': epoch,
                'losses': losses,
                'best_loss': best_loss}
        dataOut = dict(self.trainingData)
        dataOut.update(dataLocal)
        outDir = os.path.join(modeldir,model.name)
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        torch.save(model, os.path.join(outDir, 'model'+'.pth'))
        torch.save(optimizer, os.path.join(outDir, 'optimizer'+'.pth'))
        torch.save(dataOut, os.path.join(outDir, 'params'+'.pth'))
        print('new best loss: ', "{:.3e}".format(best_loss))
        return 0
