function [stftSignal, f, t] = extractSTFTFeatures(values, fs, nFFT, bound, logScale, showOutput)
    arguments
        values (1,:) double
        fs (1,1) int16
        nFFT (1,1) int16
        bound (1,1) int16 = 50
        logScale logical = false
        showOutput logical = false

    end

    % Get STFT original
    [stftOriginal, f, t] = stft(values,fs,window=hann(nFFT),FFTLength=nFFT,FrequencyRange="onesided");
    stftOriginalMag = abs(stftOriginal);
    stftOriginalMag(stftOriginalMag<0.1) = 0;
    stftOriginalMag = movmean(stftOriginalMag,3);

    stftSignal = stftOriginalMag(1:bound,:);
    
    if (logScale)
        % Log scale
        stftLog = log(stftOriginalMag+1e-10);
        stftLog = stftLog - min(min(stftLog));

        stftSignal = stftLog;
    end

    if (showOutput)
        figure()
        [ff,tt] = meshgrid(t,f);
        mesh(tt(1:bound,:),ff(1:bound,:),stftOriginalMag(1:bound,:));
        title("STFT original");

        if (logScale)
            figure()
            mesh(tt(1:bound,:),ff(1:bound,:),stftLog(1:bound,:));
            title("STFT Log");
        end
    end
end
