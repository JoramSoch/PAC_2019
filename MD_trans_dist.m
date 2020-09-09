function xt = MD_trans_dist(x, y)
% _
% Distributional Transformation
% FORMAT xt = MD_trans_dist(x, y)
% 
%     x  - source data, data to be transformed
%     y  - reference data, target for transformation
% 
%     xt - transformed data, x mapped to y
% 
% FORMAT xt = MD_trans_dist(x, y) transforms the distribution of x to the
% distribution of y, such that the empirical cumulative distribution
% function (eCDF) of x matches that of y.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Date  : 11/08/2020, 14:25


% calculate CDFs
[f1, x1] = ecdf(x);
[f2, x2] = ecdf(y);

% transform x
xt = zeros(size(x));
for i = 1:numel(x)
    j1 = find(x1==x(i));
    j1 = j1(end);
    [m, j2] = min(abs(f2-f1(j1)));
    xt(i) = x2(j2);
end;
clear m j1 j2