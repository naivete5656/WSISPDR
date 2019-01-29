function [output] = normalize(input)
%Normalize the input into [0,1] range

if isempty(input)
    output = [];
    return;
end

idx = isnan(input);



minval = min(input(:));
maxval = max(input(:));

if minval == maxval
    output = zeros(size(input));
    return;
end

output = (input-minval)/(maxval-minval);
output(idx) = nan;

output=sqrt(2*output-output.^2);