function extracted = extractTKEOFeatures(val, fs, newFs)
    arguments
        val (:,1) double
        fs (1,1) double
        newFs (1,1) double = 400
    end

    % Butterworth 4-order lowpass filter on signal
    fc1 = 75;
    cutoff = fc1/(fs/2);
    [b,a]= butter(2, cutoff, "low");
    val_LP = filter(b,a,val);

    % Calculate TKEO
    squared = val_LP(2:end-1).^2;
    before = val_LP(1:end-2);
    after = val_LP(3:end);
    val_TKEO = squared + before.*after;

    % Smooth out the TKEO
    fc = 20;
    cutoff = fc/(fs/2);
    [b,a]= butter(3, cutoff, "low");
    val_TKEO_LP = filter(b,a,val_TKEO);

    % Resample
    extracted = resample(val_TKEO_LP,newFs,fs);
end