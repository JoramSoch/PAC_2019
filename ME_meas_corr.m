function [R2, R2_adj, r, r_SC, MAE, RMSE, Obj2] = ME_meas_corr(y, y_est, p)
% _
% Measures of Correlation
% FORMAT [R2, R2_adj, r, r_SC, MAE, RMSE, Obj2] = ME_meas_corr(y, y_est, p)
% 
%     y      - true values
%     y_est  - predicted values
%     p      - number of regressors
% 
%     R2     - coefficient of determination
%     R2_adj - adjusted coefficient of determination
%     r      - Pearson correlation
%     r_SC   - Spearman correlation
%     MAE    - mean absolute error
%     RMSE   - root mean squared error
%     Obj2   - Objective 2 of PAC 2019
% 
% FORMAT [R2, R2_adj, r, r_SC, MAE, RMSE, Obj2] = ME_meas_corr(y, y_est, p)
% calculates various measures of correlation between true values y and
% predicted values y_est, obtained using p predictors.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Date  : 11/08/2020, 14:58


% prepare R-squared
RSS  = sum((y-y_est).^2);
TSS  = sum((y-mean(y)).^2);
df_r = numel(y)-p-1;
df_t = numel(y)-1;

% calculate measures
R2     = 1 - RSS/TSS;
R2_adj = 1 - (RSS/df_r)/(TSS/df_t);
r      = corr(y, y_est);
r_SC   = corr(y, y_est, 'type', 'Spearman');
MAE    = mean(abs(y-y_est));
RMSE   = sqrt(RSS/numel(y));
Obj2   = corr(y, y-y_est, 'type', 'Spearman');