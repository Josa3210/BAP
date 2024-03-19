function [output] = TKEO(val)
    squared = val(2:length(val)-1).^2;
    negOffset = val(1:length(val)-2);
    posOffset = val(3:length(val));
    output = squared - (negOffset .* posOffset);
    % ex = [ex(1); ex; ex(length(sig)-2)]; %make it the same length
end

