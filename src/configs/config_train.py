import os

includeMMBG         = False
includeLip          = True
LoadPretrainedModel = True
Monotonicity        = False
NormalizeBasisSets  = False  # normalizes all basis sets so that highest peak is 1. LEAVE THIS AT FALSE!!!
ReduceSmallMMs      = False  # Removes MMs with small amplitude to speed up training
trainLs             = True  # train the network for lipid suppresion (otherwise it's just denoising)
plotSpectraDuringTraining = True

epochs          = 100000
lr              = 6e-5
batch_size      = 64    # will be multiplied by n_bvals
plot_loss_every = 1000    # plot and print info every n epochs
window_for_current_loss = 400

modeldir        = 'trained_models/L2' # save model in
pretrained_path = os.path.join(modeldir, 'Unet021124') # load this model

bvals = range(1,1+1)