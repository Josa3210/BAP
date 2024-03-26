function extracted = extractFeatures(filepath)
    % Open the sound
    [val, fs] = audioread(filepath);

    % We want to get the same size for every sample: 4 seconds
    % get 2 seconds of points with sample rate fs
    points_2s = 2*fs;
    points_4s = 2 * points_2s;

    if (2 * points_2s) + 1 <= length(val);
        % Get the middle point of the sample
        middle_point = floor(length(val) / 2);
        val = val(middle_point - points_2s:middle_point + points_2s);
    else
        offset = points_4s + 1 - length(val);
        zero = zeros(offset,1);
        val = [val; zero];
    end

    % Filter the sound a first time using a butterworth lowpass filter.
    fc = 50;
    [b,a]= butter(4,fc/(fs/2),"low");
    val_LP = filter(b,a,val);

    % Transform the converted data using the TKEO operator
    val_LP_TKEO = TKEO(val_LP);

    % Smooth out the TKEO filtered data
    fc = 20;
    [b,a]= butter(4,fc/(fs/2),"low");
    val_LP_TKEO_LP = filter(b,a,val_LP_TKEO);


    % Normalize the data
    val_LP_TKEO_LP_norm = (val_LP_TKEO_LP - mean(val_LP_TKEO_LP))./std(val_LP_TKEO_LP);

    % if logical(showImage)
        % figure();
        % subplot(2,1,1);
        % plot(val); axis padded; title("Test sound");

        % subplot(2,1,2);
        % plot(val_LP_TKEO_LP_norm);
        % axis padded; title("sound TKEO LP norm");
    % end

    extracted = val_LP_TKEO_LP_norm;
end