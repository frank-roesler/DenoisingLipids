% resample basis sets
addpath(genpath('f:\cubric_sync\#Measurements\###NEW_DWMRS_PIPELINE\extSrc\fidA\'));

fsNew    = 2500;                       % [Hz]
noPtsNew = 406; 

inputPath  = 'f:\cubric_sync\backup_denoising\2023_05_24\mrsDenoisingV02\03_SLOW\basisSetsOld\';
outputPath = 'f:\cubric_sync\backup_denoising\2023_05_24\mrsDenoisingV02\03_SLOW\basisSetsRs\';

bsFiles = dir([inputPath '\*.mat']);

for bsFileIdx = 1:length(bsFiles)
    oldData = load([bsFiles(bsFileIdx).folder filesep bsFiles(bsFileIdx).name]);
    
    fsOld    = oldData.exptDat.sw_h; % [Hz]
    tOld     = (0:1:(length(oldData.exptDat.fid)-1))' * (1/fsOld);
    tNewMax  = ( 0:1:round((length(oldData.exptDat.fid)-1)*(fsNew/fsOld)) )' * (1/fsNew);

    fidRs = interp1( tOld, oldData.exptDat.fid, tNewMax, 'pchip' );
    
    exptDat = oldData.exptDat;
    exptDat.sw_h   = fsNew;
    exptDat.nspecC = noPtsNew;
    exptDat.fid    = fidRs(1:noPtsNew);

    save([outputPath filesep bsFiles(bsFileIdx).name], 'exptDat');
end

figure;
hold on;
plot( tOld,    real(oldData.exptDat.fid) );
plot( tNewMax(1:noPtsNew), real(exptDat.fid) );
hold off;

naa = load('f:\cubric_sync\backup_denoising\2023_05_24\mrsDenoisingV02\03_SLOW\basisSetsRs\NAA.mat')
cr = load('f:\cubric_sync\backup_denoising\2023_05_24\mrsDenoisingV02\03_SLOW\basisSetsRs\Cr.mat')

figure;
hold on
plot( real(fftshift(ifft( (naa.exptDat.fid*11 + cr.exptDat.fid*3).*exp(-pi*5*tNewMax(1:noPtsNew)), [], 1 ),1 ) ));
plot( -imag(fftshift(ifft( (naa.exptDat.fid*11 + cr.exptDat.fid*3).*exp(-pi*5*tNewMax(1:noPtsNew)), [], 1 ),1 ) ));
hold off

% testing shape of lipid basis
lip = load('f:\cubric_sync\backup_denoising\2023_05_24\mrsDenoisingV02\03_SLOW\lipidModel\lipMdl.mat')

tAx = ( 0:1:(lip.lipidModel.smpPts-1) ) * (1/lip.lipidModel.smpFrq);

constL    = pi;                        % constant needed to align with FiTAID (not 100% sure why)
constG    = (2*pi/sqrt(16*log(2)));    % constant needed to align with FiTAID (not 100% sure why)

voigtFunc = @(para,tAx) para(1,:).*exp( -1i*para(2,:).*tAx'+1i*para(3,:) ).*exp( -(constL*para(4,:)).*tAx'-(constG*para(5,:)).^2.*(tAx.^2)');

idx = 141;

para = []
para(1,:) = lip.lipidModel.lipPara{idx}.amp
para(2,:) = 2*pi*lip.lipidModel.lipPara{idx}.frq
para(3,:) = lip.lipidModel.lipPara{idx}.ph + deg2rad(51.87)
para(4,:) = lip.lipidModel.lipPara{idx}.fL
para(5,:) = 4.62

y = voigtFunc(para, t )

figure;
hold on
plot( real(fftshift(ifft( sum(y,2), [], 1 ),1 ) ));
plot( imag(fftshift(ifft( sum(y,2), [], 1 ),1 ) ));
hold off
