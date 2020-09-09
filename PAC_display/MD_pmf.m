function pmf = MD_pmf(y, x)
% _
% Probability Mass function
% FORMAT pmf = MD_pmf(y, x)
% 
%     y   - an n x 1 vector of data points
%     x   - a  1 x m vector of bin centers
% 
%     pmf - a  1 x m vector of probabilities
% 
% FORMAT pmf = MD_pmf(y, x) computes the empirical probability mass
% function (PMF) of the data in y using the bins in x.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Date  : 19/08/2020, 15:16


% estimate PMF
af  = hist(y, x);
pmf = af./numel(y);
clear af