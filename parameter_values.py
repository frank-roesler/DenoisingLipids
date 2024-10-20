from pathlib import Path

mmbg_path  = Path('data/basisSets/MMBG/MMBG_050_woCrCH2.mat')  # path to the macromolecular model (from FiTAID, voigt lines), usually not required for TE > 60ms
lip_path   = Path('data/03_SLOW/lipidModel/')  # path to the lipid model (voigt line model from svd)
metab_path = Path('data/03_SLOW/basisSetsRs/')  # path to MARSS basis sets (no pre line broadening)

noSmp       = 406         # [pts] number of sampling points (SLOW: 406)
bw          = 2500        # [Hz] sampling frequency of the ADCs (STEAM: 8000; sLASER: 5000; SLOW: 2500)

kwargs_BS = {'bw':          bw,
             'noSmp':       noSmp,
             'bsPhase':     90,               # [Deg] phase correction term for basis sets (the basis sets not necessarily have the right 0 order phase, this term is an estimate to phase the basis sets correctly (STEAM: 40; SLOW: 90))
             'lWidth':      (3,7),            # [min,max] [Hz,Hz] Lorentz Width (MMBG: [10 20]; Metab: [2.0 8.0])
             'gWidth':      (6,9),            # [min,max] [Hz,Hz] Gauss Width (equal for MMBG and Metab, default [1.5 5.0])
             'phaseOffs':   (-20,+20),        # [min,max] [Deg,Deg] phase variation
             #'phaseOffs':   (0,0),        # [min,max] [Deg,Deg] phase variation
             'freq_offset': (-2,+2),          # [min,max] [Hz] frequ offset, TODO: this should rather be a global paramter, the same for all metabolites (local frequency offsets, might be included, but should rather be small [-1...+1]Hz)
             #'noiseLvl':    (0.010,0.020)     # relative noise level 0 to 1 (SNR=inf,SNR=1)
             'noiseLvl':    (0.010,0.015)     # relative noise level 0 to 1 (SNR=inf,SNR=1)
             }

# Mekle, R., Mlynárik, V., Gambarota, G., Hergt, M., Krueger, G., & Gruetter, R. (2009). MR spectroscopy of the human brain with enhanced signal intensity at ultrashort echo times on a clinical platform at 3T and 7T. Magnetic Resonance in Medicine, 61(6), 1279–1285. https://doi.org/10.1002/mrm.21961
metab_con = { 'Asc':  (0.1 , 0.1),
              'Asp':  (2.9 , 0.5),
              'Cr':   (5.0 , 0.3),
              'GABA': (1.3 , 0.2),
              'Gln':  (2.2 , 0.4),
              'Glu':  (9.9 , 0.9),
              'Gly':  (0.3 , 0.2),
              'GPC':  (0.8 , 0.1),
              'GSH':  (1.3 , 0.2),
              'Lac':  (0.7 , 0.1),
              'mI':   (5.7 , 0.5),
              'NAA':  (11.8, 0.2),
              'NAAG': (1.1 , 0.1),
              'PCh':  (0.5 , 0.1),
              'PCr':  (3.0 , 0.3),
              'PE':   (2.5 , 0.3),
              'sI':   (0.3 , 0.2),
              'Tau':  (1.5 , 0.3),
            }

# Lipid parameters (taking voigt data from the SVD)
kwargs_Lipid = { 'bw':          bw,
                'noSmp':       noSmp,
                'freqOffs':    (-1,+1),        # random variation in lipid frequencys in each of the voigt basis functions
                'globalAmp':   (0.0,8.0),   # [mean, variance] folded normal distribution (1.0 means at the level of metabolites, this should reflect how many lipids we are seeing per pixel)
                'sdAmp':       0.1,          # relative value, random per resonance in the range (to add some variation in the lipid model)
              }

# MM parameters (taking data from voigt model in FitAID)
kwargs_MM = {'bw':          bw,
             'noSmp':       noSmp,
             'globalAmp':   0.5,   # D050: 0.5,  D100: 0.4,  D250: 0.3
             'globalL':     0,
             'sdGlobalL':   0,
             'sdMMAmp':     0.1,   # relative
             'sdPhase':     20,
             'sdFreq':      0.2,   # absolute !!![ppm]!!!
             }