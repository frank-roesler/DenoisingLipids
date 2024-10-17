import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand, randn
import torch
from numpy.fft import ifft, fftshift
import scipy.io
from glob import glob
import scipy.io
from numba import njit

def build_ppmAx(bw, noSmp):
    gamma = 42.577  # [MHz/T]
    #Bo = 2.89  # [T], old data, 4Pontus: we are now at 7T (i.e. 6.98T in reality)
    Bo = 6.98
    wCenter = 4.70  # [ppm] Center frequency of Water
    fL = Bo * gamma
    fAx = np.arange(-bw / 2 + bw / (2 * noSmp), (bw / 2) - bw / (2 * noSmp), bw / (noSmp + 1))  # [Hz]
    ppm = fAx / fL
    ppm = ppm + wCenter
    return ppm, fAx, wCenter, fL

def voigtFuncLip(para, tAx):
    tAx = tAx[:,np.newaxis]
    constL = np.pi
    constG = 2 * np.pi / np.sqrt(16 * np.log(2))
    # voigtFunc = @(para,tAx) para(1)*exp( -1i*para(2)*tAx+1i*para(3) ).*exp( -(constL*para(4))*tAx-(constG*para(5))^2*tAx.^2); # the one from MatLab
    fid = para['amplitude'] * np.exp( -1j*2*np.pi*para['freq_offset']*tAx + 1j*para['phase_offset'] ) # here we have a factor of 2*pi to align with data from SVD
    damp = np.exp(-(constL * para['lorentz_width']) * tAx - (constG*para['gauss_width']) ** 2 * tAx ** 2 )
    return fid * damp

def voigtFuncMmbg(para, tAx):
    tAx = tAx[:,np.newaxis]
    constL = np.pi
    constG = 2 * np.pi / np.sqrt(16 * np.log(2))
    fid = para['amplitude'] * np.exp(-1j*np.deg2rad(para['freq_offset']*360)*tAx + 1j*np.deg2rad(para['phase_offset']))
    damp = np.exp(-(constL * para['lorentz_width']) * tAx - (constG * para['gauss_width']) ** 2 * tAx ** 2)
    return fid * damp

def metabFunc_vec(para, tAx, y_metab):
    """Adds Gauss and Lorentz damping, frequency and phase shift to the signal y_metab for the metabolite metab_name"""
    #print( y_metab.shape[1] )
    n_metabs = y_metab.shape[1]

    amplitudes = np.zeros(n_metabs)
    #print(n_metabs)
    n_nonzero = np.random.randint(1, n_metabs+1)
    nonzero_amplitudes = np.random.choice(np.arange(n_metabs), n_nonzero)
    amplitudes[nonzero_amplitudes] = np.random.rand(n_nonzero)
    amplitudes=np.ones(n_metabs);

    constL     = np.pi
    constG     = 2 * np.pi / np.sqrt(16 * np.log(2))

    fid        = amplitudes * y_metab * para['ampl_fluc']

    phase      = np.exp( 1j*np.deg2rad(para['phaseOffs']))
    freq_shift = np.exp(-1j*np.deg2rad(para['freq_offset']*360)*tAx[:,np.newaxis])
    damp       = np.exp(-(constL*para['lorentz_width'])*tAx[:,np.newaxis] - (constG*para['gauss_width']*tAx[:,np.newaxis])**2)
    return fid * damp * phase * freq_shift

class Metab_basis:
    def __init__(self, path, kwargs_BS, metab_con, normalize_basis_sets=False):
        #print(path)
        #print(glob(path+'/*.mat'))
        metab_con_s = dict(sorted(metab_con.items(), key=lambda item: item[1]))
        self.maxMetabCon = (metab_con_s[list(metab_con_s)[-1]])[0]

        self.kwargs      = kwargs_BS
        self.metab_paths = sorted(glob(path+'/*.mat'), key = lambda s: s.casefold())
        metab_names = [metab_path.split('\\')[-1] for metab_path in self.metab_paths]   # find all *.mat files
        metab_names = sorted([name.split('.')[0] for name in metab_names], key = lambda s: s.casefold())
        self.metab_names    = metab_names
        self.naked_patterns, self.metab_sd = self.make_patterns(normalize=normalize_basis_sets, metabCon = metab_con)

    def make_patterns(self, normalize=False, metabCon = None):
        t = np.arange(self.kwargs['noSmp'])/self.kwargs['bw']
        naked_patterns = np.zeros((self.kwargs['noSmp'],len(self.metab_names)), dtype=np.complex128)
        metab_sd = np.zeros((len(self.metab_names),1), dtype=np.double)
        ctr=0
        for matPath, name in zip(self.metab_paths, self.metab_names):
            mat = scipy.io.loadmat(matPath)
            
            if metabCon:
                mrsSet = mat['exptDat'][0][0][3].squeeze() * ( (metabCon[name])[0]/self.maxMetabCon )#* np.exp(np.pi * t) # for old MARSS, where 1Hz Lorentz was added automatically and needed to be removed
                metabSd = (metabCon[name])[1]/(metabCon[name])[0]   # AD
            else:
                mrsSet = mat['exptDat'][0][0][3].squeeze()
                metabSd = 0

            naked_patterns[:,ctr] = mrsSet
            metab_sd[ctr] = metabSd
            ctr+=1

        if normalize:
            tmp = ifft(np.conj(naked_patterns),axis=0)
            max_values = np.max(np.abs(tmp),axis=0)
            naked_patterns /= max_values

        return naked_patterns, metab_sd

class Lip_basis:    # class to load the lipid model from multiple matlab files (so that we do not have to access the files in every training loop)
    def __init__(self, path, kwargs_Lip):
        self.kwargs    = kwargs_Lip
        self.lip_path  = sorted(glob(path+'/*.mat'), key = lambda s: s.casefold())
        self.lipModel = self.load_para()
    
    def load_para(self):
        t = np.arange(self.kwargs['noSmp'])/self.kwargs['bw']
        #for idx, matPath, matName in enumerate( zip( self.lip_path, self.lipid_setFiles ) ):
        startIdx = 0
        lipModel = {}
        for matFile in self.lip_path:
            matStruct = scipy.io.loadmat(matFile)
            fG = matStruct['lipidModel']['fG'][0][0][0][0]
            for idx, lip_settings in enumerate( matStruct['lipidModel']['lipPara'][0][0][0], startIdx ):
                #lip_settings[0][0]['frq'][:]
                voigtPara = {}
                voigtPara['amplitude']     = lip_settings[0][0]['amp'][:]
                voigtPara['freq_offset']   = lip_settings[0][0]['frq'][:]  
                voigtPara['phase_offset']  = lip_settings[0][0]['ph'][:]
                voigtPara['lorentz_width'] = lip_settings[0][0]['fL'][:]
                voigtPara['gauss_width']   = fG
                lipModel[idx]              = voigtPara
                #voigtFuncLip(voigtPara,np.transpose( t ))
            startIdx = idx + 1
        return lipModel

class MMBG_basis:
    def __init__(self, mmbg_path, kwargs_MM, reduce_small_mm=False):
        self.adcNAA = 1e-4  # mm²/s
        self.bRef = 200  # s/mm²
        matMMBG = scipy.io.loadmat(mmbg_path)
        MMBGpara = matMMBG['mmPara'].squeeze()
        MMBGpara_list = [[t[0].item(),t[1].item()] for t in MMBGpara]
        MMBGpara_arr = np.array(MMBGpara_list)
        if reduce_small_mm:
            idx = MMBGpara_arr[:,0] >= 2
            MMBGpara_arr = MMBGpara_arr[idx,:]


        globalPara = {}
        globalPara['freq_offset']   = matMMBG['globalPara'][0][0][1]
        globalPara['phase_offset']  = matMMBG['globalPara'][0][0][2]
        globalPara['lorentz_width'] = matMMBG['globalPara'][0][0][3]
        globalPara['gauss_width']   = matMMBG['globalPara'][0][0][4]

        self.globalPara = globalPara
        self.kwargs = {**kwargs_MM,
                     'globalPara': globalPara,
                     # 'mmPara': MMBGpara,
                     'mmPara': MMBGpara_arr,     # for vectorized version of add_mmbg
                     }


def simulate_diffusion(n_bvals, metab_basis, mmbg_basis, lip_basis, 
                       mmbg=False, lip=True,
                       normalization=None, reduce_mms=False, monotone_diffusion=False):
    """simulates MR spectrum as linear combination of basis sets with varying widths"""
    t = np.arange(metab_basis.kwargs['noSmp']) / metab_basis.kwargs['bw']
    y = metab_basis.naked_patterns  # dimensions: (noSmp, #metabolites)
    n_metabs = y.shape[1]
    para = {}
    para['gauss_width']   = metab_basis.kwargs['gWidth'][0]      + (metab_basis.kwargs['gWidth'][1]      - metab_basis.kwargs['gWidth'][0])      * rand()           # global parameter (for all resonances)
    para['freq_offset']   = metab_basis.kwargs['freq_offset'][0] + (metab_basis.kwargs['freq_offset'][1] - metab_basis.kwargs['freq_offset'][0]) * rand(n_metabs)   # local parameter  (for single resonances)
    para['lorentz_width'] = metab_basis.kwargs['lWidth'][0]      + (metab_basis.kwargs['lWidth'][1]      - metab_basis.kwargs['lWidth'][0])      * rand(n_metabs)   # local parameter  (for single resonances)
    para['ampl_fluc']     = metab_basis.metab_sd.squeeze()       *                                                                                 (rand(n_metabs)-0.5)+1   # local parameter  (for single resonances)
    if metab_basis.kwargs['phaseOffs'] != (0,0):
        para['phaseOffs'] = metab_basis.kwargs['phaseOffs'][0]   + (metab_basis.kwargs['phaseOffs'][1]   - metab_basis.kwargs['phaseOffs'][0])   * rand() + metab_basis.kwargs['bsPhase']   # global parameter (for all resonances)
    else:
        para['phaseOffs'] = 0.0 + metab_basis.kwargs['bsPhase']

    # save the global random parameters, as we need those later to get gauss and phase correct
    globPara = {}
    globPara['gauss_width'] = para['gauss_width']
    globPara['phase_offs']  = para['phaseOffs']

    y = metabFunc_vec(para, t, y)
    y = np.stack([y]*n_bvals, axis=1) # (4096, 6, 23)

    if monotone_diffusion:
        amplitudes = np.sort(np.random.rand(n_bvals,n_metabs),0)
        amplitudes = amplitudes[::-1,:]
    else:
        #amplitudes = np.random.rand(n_bvals, n_metabs)            # CHANGE BACK!!!
        amplitudes = np.random.rand(n_bvals,1)*np.full( (n_bvals, n_metabs), 1)           # AD: homogeneous signal amplitude variation (we only change individual metabs depending on the SD defined in parameter_values, but not totally randomly)
        #amplitudes = np.full( (n_bvals, n_metabs), 1)            # AD: for testing, remove signal variation

    amplitudes_mean = np.mean(amplitudes,1)
    y = amplitudes*y
    y = np.sum(y, 2)

    if mmbg:
        y = add_mmbg_diffusion(y, **mmbg_basis.kwargs, n_bvals = n_bvals, reduce_mms = reduce_mms, monotone_diffusion=monotone_diffusion)
    if lip:
        y, lipSig = add_lip(y, lip_basis, metab_basis.kwargs, globPara)
        lipSig = fftshift(ifft(np.conj(lipSig), axis=0), axes=0)
    else:
        lipSig = 0
    
    y      = fftshift(ifft(np.conj(y),      axis=0), axes=0)
    if normalization=='max_1':
        max_value = np.max(np.abs(y))
        if not max_value==0:
            y      = y/max_value
            lipSig = lipSig/max_value
    elif isinstance(normalization,float) or isinstance(normalization,int):
        y      = y/normalization
        lipSig = lipSig/normalization

    return y, amplitudes_mean, lipSig


def make_batch_diffusion(batch_size, n_bvals, metab_basis, mmbg_basis, lip_basis,
                         restrict_range=None, 
                         include_mmbg = False, # default we are at long TE mostly, so no MMBG required
                         include_lip  = True,  # default we see lipids in MRSI
                         normalization='max_1',
                         monotone_diffusion=False, **kwargs_BS):
    """makes batch for diffusion fit. Returns simulated noisy signal and corresponding noise"""
    batch = []
    #noise_std    = kwargs_BS['noiseLvl'] * np.random.rand(batch_size)
    noise_std    = kwargs_BS['noiseLvl'][0] + (kwargs_BS['noiseLvl'][1] - kwargs_BS['noiseLvl'][0]) * rand(batch_size)

    for i in range(batch_size):
        #print( metab_basis )
        #print( mmbg_basis )
        diff_signals, amplitudes_mean, lipSig = simulate_diffusion(n_bvals, metab_basis, mmbg_basis, lip_basis, 
                                                                   mmbg = include_mmbg,
                                                                   lip  = include_lip,
                                                                   normalization=normalization, monotone_diffusion=monotone_diffusion)
        
        # add noise
        noise_diff = noise_std[i] * (np.random.randn(*diff_signals.shape) + np.random.randn(*diff_signals.shape)*1j)
        diff_signals = diff_signals + noise_diff
        if normalization=='max_1':
            if not np.max(np.abs(diff_signals))==0:
                diff_signals = diff_signals/np.max(np.abs(diff_signals))
                lipSig       = lipSig/np.max(np.abs(diff_signals))
            elif isinstance(normalization, float) or isinstance(normalization, int):
                diff_signals = diff_signals / normalization
                lipSig       = lipSig / normalization
        if restrict_range:
            a,b = restrict_range
            diff_signals = diff_signals[a:b,:]
            lipSig       = lipSig[a:b,:]
            noise_diff   = noise_diff[a:b,:]

        diff_signals = torch.from_numpy(diff_signals).cfloat()
        diff_signals = torch.stack([torch.real(diff_signals),torch.imag(diff_signals)])

        if include_lip:
            lipSig = torch.from_numpy(lipSig).cfloat()
            lipSig = torch.stack([torch.real(lipSig),torch.imag(lipSig)])

        noise_diff = torch.from_numpy(noise_diff).cfloat()
        noise_diff = torch.stack([torch.real(noise_diff), torch.imag(noise_diff)])

        #batch.append((diff_signals, noise_diff))
        batch.append((diff_signals, noise_diff, lipSig))

    signal_batch = torch.stack([s for (s,g,c) in batch])
    noise_batch  = torch.stack([g for (s,g,c) in batch])
    if include_lip:
        lip_batch    = torch.stack([c for (s,g,c) in batch])
        lipOut = lip_batch.detach()
    else:
        lipOut = 0

    # TODO: we have to return the pure lipid signals as well for training lipid removal
    return signal_batch.detach(), noise_batch.detach(), lipOut

def add_lip(y, lip_basis, metab_basis_settings, globPara):
    """add lipid basis function with some random variation"""
    tAx = np.arange(metab_basis_settings['noSmp']) / metab_basis_settings['bw']

    gamma = 42.577  # [MHz/T]
    Bo    = 6.98    # [T]
    fL    = Bo * gamma  # [MHz]

    # create random parameters
    # select random lipid basis from data
    lipMdlIdxs   = np.random.randint(0, len(lip_basis.lipModel), y.shape[1] )

    globAmpFac = lip_basis.kwargs['globalAmp'][1] * np.random.randn(y.shape[1]) + lip_basis.kwargs['globalAmp'][0]
    globAmpFac[globAmpFac<0] = 0

    fidSumOut   = np.zeros((y.shape[0],y.shape[1]), dtype=np.complex128)
    # TODO: AD it would be much more efficient to do this with matrix operations instead of a for loop (needs to be fixed later)
    for idx, lipMdlIdx in enumerate( lipMdlIdxs ):
        libBasePara = lip_basis.lipModel[ lipMdlIdx ]

        # create random freq offsets [Hz] for each resonance
        voigtPara = {}
        voigtPara['amplitude']     = (1 + lip_basis.kwargs['sdAmp'] * rand( len( libBasePara['amplitude'] ) ) ) * libBasePara['amplitude'].squeeze()
        #voigtPara['amplitude']     = voigtPara['amplitude']/np.sum(voigtPara['amplitude'])
        voigtPara['freq_offset']   = libBasePara['freq_offset'].squeeze() + lip_basis.kwargs['freqOffs'][0] + (lip_basis.kwargs['freqOffs'][1] - lip_basis.kwargs['freqOffs'][0]) * rand( len( libBasePara['freq_offset'] ) )
        voigtPara['phase_offset']  = libBasePara['phase_offset'].squeeze() + np.deg2rad(globPara['phase_offs']) - np.deg2rad(metab_basis_settings["bsPhase"])   # in the lipid basis we have the "right" phase, i.e., the phase from the measurement which might be different from the basis set phasae
        voigtPara['lorentz_width'] = libBasePara['lorentz_width'].squeeze()
        voigtPara['gauss_width']   = globPara['gauss_width']*np.full( len( libBasePara['amplitude'] ), 1 )

        fid        = voigtFuncLip(voigtPara, tAx)   # create lipid voigt componets
        fidSum     = np.sum(fid, axis=-1)           # combine lipid voigt components
        # to save time, we are using the first fid point for normalization (so that we do not have to do the fft each loop)
        normFac    = abs(y[0,idx])/abs(fidSum[0])

        fidSumOut[:,idx] = fidSum*normFac*globAmpFac[idx]


    # DBG: for testing (compare lipid and metabolite spectra) ->
    # specLip    = fftshift(ifft(np.conj(fidSum), axis=0), axes=0)    # TODO: think if there is a better solution as it is computationally expansive doing the fft in each of the batch loops
    # normOneLip = max( np.abs(specLip) )         # get the normalization factor (later used for lipid amplitude scaling)
    # specMetab = fftshift(ifft(np.conj(y.squeeze()), axis=0), axes=0)
    # normOneMetab = max( np.abs(specMetab) )
    # ppm, fAx, wCenter, fL = build_ppmAx(metab_basis_settings['bw'], metab_basis_settings['noSmp'])

    # fig, ax = plt.subplots(1,1, figsize=(10,6))
    # ax.plot(ppm, specLip.real*normFac )
    # ax.plot(ppm, specMetab.real )
    # ax.plot(ppm, specMetab.real + specLip.real*normFac)
    # ax.set_xlim(4.8, 1.0)
    # plt.pause(1)
    # plt.show(block=False)
    # <-

    return fidSumOut + y, fidSumOut # lip+metab, lip only



def add_mmbg_diffusion(signal, noSmp, bw, globalPara, mmPara, n_bvals, globalAmp=1, globalL=0, sdGlobalL=3,
                       sdMMAmp=0.05, sdPhase=20, sdFreq=0.2, reduce_mms = False, monotone_diffusion=False):
    """adds macromolecular background to metab signal. Models diffusion decay with n_bvals measurements"""
    tAx = np.arange(noSmp) / bw
    gamma = 42.577      # [MHz/T]
    #Bo = 2.89  # [T], old data, 4Pontus: we are now at 7T (i.e. 6.98T in reality)
    Bo    = 6.98        # [T]
    fL    = Bo * gamma  # [MHz]
    num_mmbg = mmPara.shape[0]
    mmPara = mmPara.T
    if reduce_mms:
        idx = mmPara[0,:] >= 1
        mmPara = mmPara[:,idx]
        num_mmbg = mmPara.shape[1]
        print(num_mmbg)
    paraVoigt = {}
    paraVoigt['phase_offset']  = sdPhase * randn()
    if globalL == 0:
        paraVoigt['lorentz_width'] = globalPara['lorentz_width'] + sdGlobalL * randn()
    else:
        paraVoigt['lorentz_width'] = globalL  + sdGlobalL * randn()
    paraVoigt['gauss_width']   = globalPara['gauss_width']
    paraVoigt['amplitude']     = mmPara[0:1,:] * (1 + sdMMAmp * randn(1,num_mmbg)) * globalAmp*rand()
    #paraVoigt['amplitude']     = mmPara[0:1,:]
    paraVoigt['freq_offset']   = mmPara[1:2,:] + sdFreq*fL * randn(1,num_mmbg)
    #paraVoigt['freq_offset']   = mmPara[1:2,:]
    mmbg_peaks = voigtFuncMmbg(paraVoigt, tAx)
    mmbg_peaks = np.stack([mmbg_peaks] * n_bvals, axis=1)  # (4096, 6, 51)
    if monotone_diffusion:
        diff_amplitudes = np.sort(rand(n_bvals, num_mmbg), 0)
        diff_amplitudes = diff_amplitudes[::-1, :]
    else:
        diff_amplitudes = rand(n_bvals, num_mmbg)
    mmbg_peaks = diff_amplitudes * mmbg_peaks
    signalOut = np.sum(mmbg_peaks, axis=-1)
    signalOut = signalOut.squeeze() + signal
    return signalOut







# # -------------------------------------
# #  TEST DIFFUSION AND MMBG SIMULATION:
# # -------------------------------------
# from parameter_values import *
# metab_basis = Metab_basis(metab_path, kwargs_BS, normalize_basis_sets=False)
# mmbg_basis  = MMBG_basis(mmbg_path, kwargs_MM)
# simulator   = Simulator(metab_basis, mmbg_basis)
# batch_size = 10
# ppm, fAx, wCenter, fL = build_ppmAx(bw, noSmp)
# signal_batch, y = make_batch_diffusion(batch_size, simulator, include_mmbg=includeMMBG, normalization_before_noise='max_1',
#                          normalization_after_noise='max_1', **kwargs_BS)
# n_bvals = y.shape[-1]
# print(y.shape)
# M_mean = 0
# cmap = plt.get_cmap('winter',n_bvals)
# for B in range(batch_size):
#     fig, ax = plt.subplots(2,1, figsize=(10,6))
#     Max = np.max(torch.sqrt(y[B,0,:,0]**2 + y[B,1,:,0]**2).numpy())
#     M_mean += Max/batch_size
#     for i in range(n_bvals):
#         ax[0].plot(ppm, y[B,0,:,i], linewidth=0.5, color=cmap(i))
#         ax[1].plot(ppm, y[B,1,:,i], linewidth=0.5, color=cmap(i))
#     ax[0].set_xlim(ppmRange[1]+0.5, ppmRange[0]-0.5)
#     ax[1].set_xlim(ppmRange[1]+0.5, ppmRange[0]-0.5)
#     ax[0].set_ylim(-0.5,1)
#     ax[1].set_ylim(-0.5,1)
# plt.show()