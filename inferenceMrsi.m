%%%%%%%%%%%%%%%%%
%
% Denoise DW-MRS
%
%%%%%%%%%%%%%%%%%
clear;

anaMatrixPath = 'f:\cubric_sync\backup_denoising\2023_05_24\mrsDenoisingV02\03_SLOW\mrsiData\mrsiData_Lip.mat';
dnModelPath   = 'f:\epfl_sync\python\projects\DenoisingLipids\trained_models\DiffusionNet_compr_15x3_16x3_32\model.pth';

NoiseFit    = true;
MultiDimFit = true;

% Initialize Python interface for denoising
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 	1. get environment ready
try
    [~,hostName] = system('hostname');
    if contains(hostName, 'DESKTOP-7C8TNDJ')    % cubric computer
        %pyenv("ExecutionMode","OutOfProcess")
        pe = pyenv('Version','C:\Users\adoering\.conda\envs\matlab\python.exe','ExecutionMode','OutOfProcess');
    end
    if contains(hostName, 'DESKTOP-2ML7IQ3')                % home computer
        pe = pyenv('Version','C:\Users\adoering\.conda\envs\matlab\python.exe');
    end
    if contains(hostName, 'cibmaitpc11')                % home computer
        pe = pyenv('Version','c:\Users\adoering\AppData\Local\anaconda3\envs\ai-pytorch-proj\python.exe');
    end
catch
    pe = pyenv();
end
pyTCL = fullfile(pe.Home, 'tcl', 'tcl8.6');
pyTK  = fullfile(pe.Home, 'tcl', 'tk8.6');

setenv('TCL_LIBRARY', pyTCL);
setenv('TK_LIBRARY', pyTK);
setenv('KMP_DUPLICATE_LIB_OK','True');

insert(py.sys.path, int64(0), fileparts(matlab.desktop.editor.getActiveFilename));

py.importlib.import_module('scipy');
py.importlib.import_module('numpy');
py.importlib.import_module('torch');
py.importlib.import_module('scipy');
py.importlib.import_module('matplotlib');
py.importlib.import_module('utils_infer');
py.importlib.import_module('nets');

%   2. initialized cuda
device = py.torch.device('cuda');
if (py.torch.cuda.is_available())
    cudaAvl = 'Yes';
else
    cudaAvl = 'No';
end
disp( ['CUDA avialble: ' cudaAvl] );

%   3. load dn model
model = py.torch.load(dnModelPath, device, pyargs('weights_only',false));
model.eval();

load( anaMatrixPath );

#idxList = [ [ 5 14]; [ 5+7 14+3] ];    % occipital
idxList = [ [ 14 14]; [ 14+7 14+3] ];    % occipital

[X, Y] = meshgrid(idxList(1,1):idxList(2,1),idxList(1,2):idxList(2,2))

indices = sub2ind([32 32], Y, X);

anaImg = load('f:\cubric_sync\backup_denoising\2023_05_24\mrsDenoisingV02\03_SLOW\mrsiData\anatomImg.mat');

img = anaImg.anatomicImg;
img(indices(:)) = NaN;
figure;
axis square; %hold on;
h = imagesc( img ); %colormap jet; colorbar; axis square; hold on;
%set(h, 'AlphaData', double( brainMask ) + double( brainMask ~=1 )*0.8 );
daspect([1 1 1]);

linIdx = indices(:);

y = [];
for diffExpItx = 1:length( linIdx )
    linIdxS = linIdx(diffExpItx);
    try
        y(:,diffExpItx) = fftshift(ifft(mrsiData.fids(:,linIdxS)',[],2), 2 );
        %y(:,diffExpItx) = conj(y(:,diffExpItx));
    catch
        y(:,diffExpItx) = fftshift(ifft(mrsiData.fids(:,linIdxS)',[],2), 2 );
        %y(:,diffExpItx) = conj(y(:,diffExpItx));
    end
        
    if (diffExpItx == 1)
        y_py = py.numpy.array(num2cell(transpose(y(:,diffExpItx))), 'complex128');
    else
        y_py = py.numpy.append(y_py, py.numpy.array(num2cell(transpose(y(:,diffExpItx))), 'complex128'), int64(0) );
    end
end

y_py = py.numpy.reshape(y_py, [int64(length( linIdx )), size(y,1)] );

% figure
% plot( reshape( double( py.array.array('d',py.numpy.nditer(y_py.real)) ), [size(y,1), length( expInfo.dwMRSdata )] ) );

y_dn_cplx = py.utils_infer.denoise_signal( y_py.T, model, pyargs('diffusion', MultiDimFit, 'noise_fit', NoiseFit, 'device', device ) )

spec = double( py.array.array('d',py.numpy.nditer(y_dn_cplx.real)) ) + i*double( py.array.array('d',py.numpy.nditer(y_dn_cplx.imag)) );

spec = reshape(spec, [size(y,1), int64(length( linIdx ))] );

figure;
%hold on;
ax1 = subplot(2,1,1);
plot(fliplr(mrsiData.ppm), real( y(:,:) ) );
axHdl = gca;
axHdl.XDir = 'reverse';
ax2 = subplot(2,1,2);
plot(fliplr(mrsiData.ppm), real( spec(:,:) ));
linkaxes([ax1,ax2],'x');
axHdl = gca;
axHdl.XDir = 'reverse';
xlim([0.5 4.3]);
%plot( flip(real( getfield( op_averaging( metab ), 'specs') ) ) );
%hold off;