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

     % Get the envelopes
    maxVal = movmean(movmax(signal,900),5000);
    minVal = movmean(movmin(signal,900),5000);

    % Filter values based on enveloppe
    filtered = signal;
    filtered(maxVal < filtered) = maxVal(maxVal < filtered);
    filtered(filtered < minVal) = minVal(filtered < minVal);
    signal = filtered;

    % Extract noise in freq domain
    sNoise = stft(profile,fs,Window=window,OverlapLength=nOverlap,FFTLength=nFFT);
    sNoiseMag = abs(sNoise);

    % Extract signal in freq domain
    sSignal = stft(signal,fs,Window=window,OverlapLength=nOverlap,FFTLength=nFFT);
    sSignalMag = abs(sSignal);
    sSignalAng = angle(sSignal);
    sSignalAngInfo = exp(1j*sSignalAng);

     % Calculate average noise
    avgNoise = sum(sNoiseMag,2)/size(sNoiseMag,2);
    avgNoiseMat = avgNoise .* ones(1,size(sSignal,2));


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

    maxNoise = max(max(sNoiseMag));
    % Residual Noise Reduction
    if (residualNoiseReduction)
        nFrames = size(sExtracted,2);
        for frame = 2:nFrames-1
            if (sExtracted(frame) > 0) && (maxNoise > sExtracted(frame))
                minval = min([sExtracted(frame-1), sExtracted(frame), sExtracted(frame+1)]);
                sExtracted(frame) = minval;
            end
        end
    end

    % Reconstruct the signal
    sExtracted = sExtracted .* sSignalAngInfo;

    % Enlarge extracted signal for better reconstruction
    halfEnd = floor(size(sExtracted,2)/2);
    sExtracted = [sExtracted(:,halfEnd:end), sExtracted, sExtracted(:,1:halfEnd-1)];
   
    sFiltered = real(istft(sExtracted,fs,Window=window,OverlapLength=nOverlap,FFTLength=nFFT));
    sFiltered = sFiltered(end/4:end*3/4);

    % Calculate SNR
    noiseSNR = sqrt(sum(profile)^2/size(profile,1));
    signalSNR = sqrt(sum(signal)^2/size(signal,1));
    filteredSNR = sqrt(sum(sFiltered)^2/size(sFiltered,1));

    SNRref = 20*log(signalSNR/noiseSNR);
    SNRfilt = 20*log(filteredSNR/noiseSNR);

    SNRDiff = SNRref - SNRfilt;
end