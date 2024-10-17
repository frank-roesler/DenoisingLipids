% get lipid basis

load('/home/adoering/Desktop/cibmaitsrv1/backup/cubric_sync/backup_denoising/2023_05_24/mrsDenoisingV02/03_SLOW/lipidModel/lipMdl.mat');

tVec = (0:1:(lipidModel.smpPts-1)) * (1/lipidModel.smpFrq);
fG   = lipidModel.fG;   % can be chose according to the random fG value of the trainingsdata

% para(1):  Amplitude -> we can vary a global factor for the amplitude to increase or decrease lipids (also not clear how we scale everything in the first place)
% para(2):  frq offset -> offset chracteristic for lipids (maybe globally +- some value [but only a few Hz])
% para(3):  phase -> here we can add the random phase (+random phase, in radiant NOT degree)
% para(4):  lorentz width -> lorentz we should not touch
% para(5):  gauss with -> c.f. above

voigtFunc = @(para,tAx) para(1)*exp( -1i*para(2)*tAx+1i*para(3) ).*exp( -(constL*para(4))*tAx-(constG*para(5))^2*tAx.^2);

lsDataItx = 30; % select a random lipid pixel of your choise

% reconstruct lipids basis
lsBasisFid = [];
for compItx = 1:length( lipidModel.lipPara{1, lsDataItx}.amp )
    if compItx == 1
        lsBasisFid = voigtFunc( [lipidModel.lipPara{1, 30}.amp(compItx), ...
                                 2*pi*lipidModel.lipPara{1, 30}.frq(compItx), ...
                                 lipidModel.lipPara{1, 30}.ph(compItx), ...
                                 lipidModel.lipPara{1, 30}.fL(compItx), ...
                                 fG], ...
                                tVec' ...
                              );
    else
        lsBasisFid = voigtFunc( [lipidModel.lipPara{1, 30}.amp(compItx), ...
                                 2*pi*lipidModel.lipPara{1, 30}.frq(compItx), ...
                                 lipidModel.lipPara{1, 30}.ph(compItx), ...
                                 lipidModel.lipPara{1, 30}.fL(compItx), ...
                                 fG], ...
                                tVec' ...
                              ) + lsBasisFid;
    end
end

% we can add now this fid to the trainings data (be carefull sometimes we
% have the spectrum, sometimes the fid in the python code, if we add it to
% the spectrum we have to do the fft first)

% plot lipids spec
lsBasisSpec = fftshift(ifft(lsBasisFid,[],1), 1 ) ;

figure;
hold on;
plot( real( lsBasisSpec ) );
plot( imag( lsBasisSpec ) );
hold off;
