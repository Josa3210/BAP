function extracted = extractFeatures2(val, fs)
    arguments
        val (:,1) double
        fs (1,1) double
    end

    % Get the envelopes
    maxVal = movmean(movmax(val,900),5000);
    minVal = movmean(movmin(val,900),5000);

    % Filter values based on enveloppe
    filtered = val;
    filtered(maxVal < filtered) = maxVal(maxVal < filtered);
    filtered(filtered < minVal) = minVal(filtered < minVal);

    % Butterworth 4-order lowpass filter on signal
    fc1 = 75;
    cutoff = fc1/(fs/2);
    [b,a]= butter(2, cutoff, "low");
    val_LP = filter(b,a,filtered);

    % Calculate TKEO
    squared = val_LP(2:end-1).^2;
    before = val_LP(1:end-2);
    after = val_LP(3:end);
    val_TKEO = squared + before.*after;

    % Smooth out the TKEO
    fc = 20;
    cutoff = fc/(fs/2);
    [b,a]= butter(3, cutoff, "low");
    extracted = filter(b,a,val_TKEO);
end