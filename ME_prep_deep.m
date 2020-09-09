function [X1p, X2p] = ME_prep_deep(X1, X2)
% _
% Prepare for Deep Learning
% FORMAT [X1p, X2p] = ME_prep_deep(X1, X2)
% 
%     X1  - training design matrix
%     X2  - validation design matrix
% 
%     X1p - prepared training design matrix
%     X2p - prepared validation design matrix
% 
% FORMAT [X1p, X2p] = ME_prep_deep(X1, X2) takes design matrices X1 and X2
% and (i) removes constant columns, (ii) normalizes (z-scores) predictors
% and (iii) transforms matrices into cell arrays suitable for "trainNetwork"
% used by MATLAB's Deep Learning Toolbox [1].
% 
% References:
% [1] MathWorks (2020): "Sequence-to-Sequence Regression Using Deep Learning";
%     URL: https://de.mathworks.com/help/deeplearning/ug/sequence-to-sequence-
%     regression-using-deep-learning.html.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Date  : 13/08/2020, 12:36


% remove constant columns
incl  = true(1,size(X1,2));
for j = 1:size(X1,2)
    if min(X1(:,j)) == max(X1(:,j))
        incl(j) = false;
    end;
end;
X1 = X1(:,incl);
X2 = X2(:,incl);
clear incl

% normalize (z-score) predictors
X1z = zeros(size(X1));
X2z = zeros(size(X2));
for j = 1:size(X1,2)
    % normalize, if continuous preditors
    if numel(unique(X1(:,j)))>2
        X1z(:,j) = (X1(:,j)-mean(X1(:,j)))/std(X1(:,j),0);
        X2z(:,j) = (X2(:,j)-mean(X2(:,j)))/std(X2(:,j),0);
    % keep, if indicator variables
    else
        X1z(:,j) =  X1(:,j);
        X2z(:,j) =  X2(:,j);
    end;
end;

% transform matrices into cell arrays
X1p = cell(size(X1,1),1);
X2p = cell(size(X2,1),1);
for i = 1:size(X1,1)
    X1p{i} = X1z(i,:)';
end;
for i = 1:size(X2,1)
    X2p{i} = X2z(i,:)';
end;