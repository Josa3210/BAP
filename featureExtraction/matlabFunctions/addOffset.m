function [offsettedSignal] = AddOffset(signal, fs, maxTimeOffset)
    arguments
        signal (:,1) double
        fs (1,1) int64
        maxTimeOffset (1,1) int64
    end
    maxOffset = round(maxTimeOffset * fs); % Max offset in seconds
    kRandom = round(rand(1)*maxOffset);
    offsettedSignal = [zeros(1,kRandom), signal(1:(size(signal,1)-kRandom))'];
end
