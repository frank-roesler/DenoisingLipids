from src.configs.config_simul import *
from src.configs.config_train import *
import torch
import os
import pathlib
import platform

myPlatform = platform.system()
if myPlatform == 'Darwin':
    pathlib.WindowsPath = pathlib.PosixPath
print(myPlatform)


class Checkpoint:
    def __init__(self):
        self.trainingParams = {
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
                'losses': losses,
                'best_loss': best_loss}
        dataOut = dict(self.trainingParams)
        dataOut.update(dataLocal)
        outDir = os.path.join(modeldir,model.name)
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        torch.save(model, os.path.join(outDir, 'model'+'.pth'))
        torch.save(optimizer, os.path.join(outDir, 'optimizer'+'.pth'))
        torch.save(dataOut, os.path.join(outDir, 'params'+'.pth'))
        print('new best loss: ', "{:.3e}".format(best_loss))
        return 0

    def load_pretrained_model(self, directory_path, device):
        model = torch.load(os.path.join(directory_path, 'model.pth'), map_location=device, weights_only=False)
        optim = torch.load(os.path.join(directory_path, 'optimizer.pth'), map_location=device, weights_only=False)
        self.trainingParams = torch.load(os.path.join(directory_path, 'params.pth'), weights_only=False)

        return model, optim
