function [extracted, newFs] = extractTKEOFeatures(val, fs, newFs)
    arguments
        val (:,1) double
        fs (1,1) int32
        newFs (1,1) double = 441
    end

    % Calculate TKEO
    squared = val(2:end-1).^2;
    before = val(1:end-2);
    after = val(3:end);
    val_TKEO = [val(1); squared + before.*after ; val(end)];

    % Smooth out the TKEO
    fc = 20;
    cutoff = fc/(double(fs)/2);
    [b,a]= butter(3, cutoff, "low");
    val_TKEO_LP = filter(b,a,val_TKEO);
    % Resample
    extracted = resample(val_TKEO_LP,newFs,fs);
end