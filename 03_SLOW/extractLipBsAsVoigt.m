% store lipid characteristics

 mask.maskLipid(:)

lipidFirstHlf = op_takeaverages( tmpFirstHalf, find(mask.maskLipid(:)) );
lipidSecondHlf = op_takeaverages( tmpSecondHalf_alg, find(mask.maskLipid(:)) );


lipidFirstHlf = tmpFirstHalf;

mrsiData = tmpFirstHalfLS;
anatomicImg = squeeze( abs( dataMrsi.waterRefRecon(:,:,1) ) );

minLw    =  2; % [Hz]
maxLW    = 40;

% we have to find the optimal number of components by successive reduction
w = warning('error', 'MATLAB:DELETE:Permission');
%[ msg , warnID ] = lastwarn
warning('error', 'MATLAB:lscov:RankDefDesignMat'); % set warning to error

%ppmRange = [min(lipidFirstHlf.ppm) 5.0];
ppmRange = [1.7 4.1];
minLw    =  1; % [Hz]
maxLW    = 40;

fG = 5;

constL    = pi;                        % constant needed to align with FiTAID (not 100% sure why)
constG    = (2*pi/sqrt(16*log(2)));    % constant needed to align with FiTAID (not 100% sure why)

voigtFunc = @(para,tAx) para(1)*exp( -1i*para(2)*tAx+1i*para(3) ).*exp( -(constL*para(4))*tAx-(constG*para(5))^2*tAx.^2);

fL = @(fV,fG) [ (2*0.5346*fV + sqrt( (2*0.5346*fV)^2 - 4*(0.5346^2-0.2166)*(fV^2-fG^2) ))/(2*(0.5346^2-0.2166)) (2*0.5346*fV - sqrt( (2*0.5346*fV)^2 - 4*(0.5346^2-0.2166)*(fV^2-fG^2) ))/(2*(0.5346^2-0.2166)) ];

lipidModel = [];
for itx = 1:lipidFirstHlf.sz(2)
    itx = sub2ind([32 32], 14, 14);
    tmp_fidA = op_takeaverages( lipidFirstHlf,itx);
    
    % create loop to find best min number of SVD components
    repCon = true;
    startComp = tmp_fidA.sz(1)/2+1;    % number of HLSVD components to start with
    while repCon
        try
            %fid = svdfid( tmp_fidA.fids, 8, tmp_fidA.spectralwidth, (4.7-max(wsRange)+applPpmOffset)*tmp_fidA.txfrq, (4.7-min(wsRange)+applPpmOffset)*tmp_fidA.txfrq, -tmp_fidA.txfrq*10000, tmp_fidA.txfrq*10000, startComp,[] );
            
            [fid,p,~,Z] = svdfid( tmp_fidA.fids, 2, tmp_fidA.spectralwidth, (4.7-max(ppmRange))*tmp_fidA.txfrq, (4.7-min(ppmRange))*tmp_fidA.txfrq, -100, +tmp_fidA.spectralwidth, startComp,[] );
            
            
            % when it works leave loop
            display(['Number of components ' num2str(startComp) ' converaged.']);
            repCon = false;
        catch
            %display( ['Number of components ' num2str(startComp) ' not converaged.'] );
            startComp = startComp - 1;
            if startComp == 1
                repCon = false;
                display('Reached end without HLSVD applied.');
                % in this worst case we just keep the spectra untouched
                fid = complex( zeros( size(tmp_fidA.fids) ) );
            end
        end
    end

    ampClx = p(:,1).*exp(1i*p(:,4) );
    svdFidClx = repmat(ampClx, [1 tmp_fidA.sz(1)]).'.*Z;
    
    % list of components to remove baseline
    bsRmvIdx = (p(:,3) > minLw & p(:,3) < maxLW );

    % decomposition of svd into voigts
    res = [];
    fVtest = [];
    fvParaEnd = [];
    for pltIdx = 1:size(p,1)
            %res = voigtFunc( [p(pltIdx,1), p(pltIdx,2)+4.7*tmp_fidA.txfrq, p(pltIdx,4), min(fL(p(pltIdx,3),fG)), fG], tmp_fidA.t' );
        fVtest{pltIdx} = fL(p(pltIdx,3),fG);

        if ( sum( fVtest{pltIdx} > 0 ) == 2 )
            fvPara = min(fVtest{pltIdx});
        end
        if ( sum( fVtest{pltIdx} > 0 ) == 1 || sum( fVtest{pltIdx} > 0 ) == 0 )
            fvPara = max(fVtest{pltIdx});
        end

        fvParaEnd(pltIdx) = fvPara;

        res(:,pltIdx) = voigtFunc( [p(pltIdx,1), p(pltIdx,2)*2*pi, p(pltIdx,4), fvPara, fG], tmp_fidA.t' );
    end

    lipidModel.fG = fG;
    lipidModel.smpPts = tmp_fidA.sz(1);
    lipidModel.smpFrq = tmp_fidA.spectralwidth;
    lipidModel.lipPara{itx}.amp = p(bsRmvIdx,1);
    lipidModel.lipPara{itx}.frq = p(bsRmvIdx,2);
    lipidModel.lipPara{itx}.ph  = p(bsRmvIdx,4);
    lipidModel.lipPara{itx}.fL  = fvParaEnd(bsRmvIdx)';

        figure;
        hold on;
        %plot( tmp_fidA.ppm, real( fftshift(ifft(tmp_fidA.fids,[],1), 1 ) ) );
        plot( tmp_fidA.ppm, real( fftshift(ifft( sum( res(:,bsRmvIdx),2),[],1), 1 ) ) );
        %plot( tmp_fidA.ppm, real( fftshift(ifft( res(:,pltIdx),[],1), 1 ) ), 'k' );
        plot( tmp_fidA.ppm, real( fftshift(ifft( sum( svdFidClx(:, bsRmvIdx ),2),[],1), 1 ) ) );
        plot( tmp_fidA.ppm, real( fftshift(ifft( sum( tmp_fidA.fids(:, : ),2),[],1), 1 ) ) );
        %plot( tmp_fidA.ppm, real( fftshift(ifft(sum( svdFidClx(:, : ),2),[],1), 1 ) ) );
        %plot( tmp_fidA.ppm, real( fftshift(ifft(tmp_fidA.fids - sum( svdFidClx(:, p(:,3) < 0 | wsIdx | bsRmvIdx | spIdx ),2),[],1), 1 ) ) );
        %plot( real( fftshift(ifft( tmp_fidA.fids - sum( svdFidClx(:, bsRmvIdx | wsIdx ),2),[],1), 1 ) ) );
        hold off;
        axHdl = gca;
        axHdl.XDir = 'reverse';
        xlim([1.4 4.1]);
        % 
        % 
        % figure;
        % hold on;
        % plot( tmp_fidA.ppm, imag( fftshift(ifft(tmp_fidA.fids,[],1), 1 ) ) );
        % plot( tmp_fidA.ppm, imag( fftshift(ifft( sum( svdFidClx(:, bsRmvIdx ),2),[],1), 1 ) ) );
        % plot( tmp_fidA.ppm, imag( fftshift(ifft( sum( res(:,bsRmvIdx),2),[],1), 1 ) ) );        
        % %plot( tmp_fidA.ppm, real( fftshift(ifft(sum( svdFidClx(:, : ),2),[],1), 1 ) ) );
        % %plot( tmp_fidA.ppm, real( fftshift(ifft(tmp_fidA.fids - sum( svdFidClx(:, p(:,3) < 0 | wsIdx | bsRmvIdx | spIdx ),2),[],1), 1 ) ) );
        % %plot( real( fftshift(ifft( tmp_fidA.fids - sum( svdFidClx(:, bsRmvIdx | wsIdx ),2),[],1), 1 ) ) );
        % hold off;
        % axHdl = gca;
        % axHdl.XDir = 'reverse';

end