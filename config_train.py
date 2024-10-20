
includeMMBG         = False
includeLip          = True
LoadPretrainedModel = False
Monotonicity        = False
NormalizeBasisSets  = False  # normalizes all basis sets so that highest peak is 1. LEAVE THIS AT FALSE!!!
ReduceSmallMMs      = False  # Removes MMs with small amplitude to speed up training
trainLs             = True  # train the network for lipid suppresion (otherwise it's just denoising)
plotSpectraDuringTraining = True

epochs          = 100000
lr              = 6e-5
batch_size      = 32    # will be multiplied by n_bvals
plot_loss_every = 1000    # plot and print info every n epochs

modeldir  = 'trained_models/' # save model as
modelname = modeldir + 'DiffusionNet_compr_15x3_16x3_32' # load this model

bvals = range(32,32+1)