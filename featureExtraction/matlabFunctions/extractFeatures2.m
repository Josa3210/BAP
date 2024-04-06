function [val, extracted] = extractFeatures2(val, fs, showImage)

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

if logical(showImage)
    subplot(2,1,2);
    plot(filtered); axis padded;
    hold on
    plot(maxVal,"LineWidth",1.5);
    plot(minVal,"LineWidth",1.5);
    hold off
    title("Filtered signal plus envelopes");
    legend("signal", "maxEnvelope", "minEnvelope");

    subplot(2,1,1);
    plot(val); axis padded;
    
    hold on
    plot(maxVal,"LineWidth",1.5);
    plot(minVal,"LineWidth",1.5);
    hold off
    title("Original signal plus envelopes");
    legend("signal", "maxEnvelope", "minEnvelope");
    
    subplot(2,1,1);
    plot(val_LP);    
    axis padded; title("LP filtered signal")

    subplot(2,1,2);
    plot(val_TKEO_LP)
    axis padded;
    title("TKEO singal LP filtered")
end