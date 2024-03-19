function extracted = extractFeatures(filepath)
    % Open the sound
    [val, fs] = audioread(filepath);

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