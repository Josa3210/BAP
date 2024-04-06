function [sFiltered, SNRDiff] = spectralSubtraction(signal, profile, fs, nFFT, nFramesAveraged, overlap, residualNoiseReduction)
    arguments
        signal (:,1) double
        profile (:,1) double
        fs (1,1) int16
        nFFT (1,1) int16
        nFramesAveraged (1,1) int16
        overlap (1,1) double = 0.5
        residualNoiseReduction logical = true
    end
    
    window = hann(nFFT);
    nOverlap= floor(nFFT * overlap);

    % Extract noise in freq domain
    sNoise = stft(profile,fs,Window=window,OverlapLength=nOverlap,FFTLength=nFFT);
    sNoiseMag = abs(sNoise);

    % Calculate average noise
    avgNoise = sum(sNoiseMag,2)/size(sNoiseMag,2);
    avgNoiseMat = avgNoise .* ones(1,size(sNoise,2));

    % Extract signal in freq domain
    sSignal = stft(signal,fs,Window=window,OverlapLength=nOverlap,FFTLength=nFFT);
    sSignalMag = abs(sSignal);
    sSignalAng = angle(sSignal);
    sSignalAngInfo = exp(1j*sSignalAng);

    % Average frames
    if nFramesAveraged > 0
        sAvgdSignal = movmean(sSignalMag,nFramesAveraged);
    else
        sAvgdSignal = sSignalMag;
    end

    % Subtract noise
    sExtracted = sAvgdSignal - avgNoiseMat;
    
    % Half wave rectify
    sExtracted(sExtracted < 0) = 0;

    % Residual Noise Reduction
    if (residualNoiseReduction)
        maxNoise = max(sNoiseMag);
        nFrames = size(sExtracted,2);
        for frame = 2:nFrames-1
            if (sExtracted(frame) > 0) && (maxNoise(frame) > sExtracted(frame))
                minval = min([sExtracted(frame-1), sExtracted(frame), sExtracted(frame+1)]);
                sExtracted(frame) = minval;
            end
        end
    end

    % Reconstruct the signal
    sExtracted = sExtracted .* sSignalAngInfo;

    % Enlarge extracted signal for better reconstruction
    sExtracted = [sExtracted(:,end/2:end), sExtracted, sExtracted(:,1:end/2-1)];
   
    sFiltered = real(istft(sExtracted,fs,Window=window,OverlapLength=nOverlap,FFTLength=nFFT));
    sFiltered = sFiltered(end/4:end*3/4);

    % Calculate SNR
    noiseSNR = sqrt(sum(profile)^2/size(profile,1)); 
    signalSNR = sqrt(sum(signal)^2/size(signal,1));
    filteredSNR = sqrt(sum(sFiltered)^2/size(sFiltered,1));


    SNRref = 20*log(signalSNR/noiseSNR);
    SNRfilt = 20*log(filteredSNR/noiseSNR);

    SNRDiff = SNRfilt - SNRref;

end
