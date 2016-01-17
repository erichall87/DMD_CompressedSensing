function out=Soft_thresh(A,thresh)
%SOFT_THRESH Soft thresholding function
%
%   Takes in an array and soft thresholds according to the threshold
%   provided

if thresh<0
    error('thresh must be positive')
end

out=(A+thresh).*(A<=-thresh)+(A-thresh).*(A>=thresh);