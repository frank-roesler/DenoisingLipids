import torch
import numpy as np
from utils_info import print_training_data, InfoScreen
from utils_simul import make_batch_diffusion, MMBG_basis, Metab_basis, Lip_basis, build_ppmAx
from utils_io import Checkpoint, load_model
from nets import DiffusionNet,UNet,DiffusionNet_compr
from parameter_values import *
from config_train import *
import matplotlib.pyplot as plt


device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = DiffusionNet_compr(ks1=(15,3), ks2=(16,3), nc=32).to(device)

path = '/Users/frankrosler/Desktop/PhD/DiffusionNet_compr_15x3_16x3_32'
checkpoint = torch.load(path, map_location=device)
print(checkpoint)