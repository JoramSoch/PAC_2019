% PAC 2019: estimate model (for Frontiers 2020 re-analysis)
% _
% Predictive Analytics Competition 2019: model estimation
% 
% The deep neural network (DNN) regression in this script is inspired from:
% https://de.mathworks.com/help/deeplearning/ug/sequence-to-sequence-regression-using-deep-learning.html
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Date  : 11/08/2020, 15:59 (V1) / 13/08/2020, 13:05 (V2) / 19/08/2020, 13:43 (V3)


clear
close all

%%% Step 0: Load everything %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load data
load PAC_specify.mat
load PAC_specify_test_age.mat

% specify analyses
cols = ['grb'];
mods = {'GLM', 'SVR', 'DNN'};
meth = {'w/o TF', 'with TF'};
meas = {'R^2', 'R^2_{adj}', 'r', 'r_{SC}', 'MAE', 'RMSE', 'Obj. 2'};
M = numel(mods);                % number of models
K = numel(meth);                % number of methods
S = 10;                         % number of subsets, s.t. S|n1

% define indices
j2d = 1;                        % model to display
k2d = 2;                        % method to display
% h - CV folds / i - subjects / j - models / k - methods

% specify histograms
dy    =  2.5;
y_bin = [15+(dy/2):dy:90-(dy/2)];

% specify DNN layers
layers = [...
    sequenceInputLayer(250)
    lstmLayer(125, 'Output', 'Last')
    fullyConnectedLayer(50)
    dropoutLayer(0.5)
    fullyConnectedLayer(1)
    regressionLayer];

% specify DNN options
maxepochs = 100;
plotting = 'none'; % 'training-progress';
options = trainingOptions('adam', ...
    'MaxEpochs',maxepochs, ...
    'MiniBatchSize',20, ...
    'InitialLearnRate',0.01, ...
    'GradientThreshold',1, ...
    'Shuffle','never', ...
    'Plots',plotting, ...
    'Verbose',0);


%%% Step 1: Estimate model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% assemble designs
n1 = numel(sID1);
X1 = [GM1, WM1, c1]; % X1 = [GM1, WM1, c1(:,2:end), ones(n1,1)];
p1 = size(X1,2);
n2 = numel(sID2);
X2 = [GM2, WM2, c2]; % X2 = [GM2, WM2, c2(:,2:end), ones(n2,1)];
p2 = size(X2,2);

% partition training data
npS =  n1/S;
i1  = [1:n1];
i1a = cell(S,1);
i1b = cell(S,1);
for h = 1:S
    i1b{h} = [((h-1)*npS+1):(h*npS)];       % tuning set = current fold
    i1a{h} = setdiff(i1,i1b{h});            % training set = all other folds
end;

% preallocate predictions
y1_est  = zeros(n1,2,M);
y1b_est = zeros(n1,2,M);
y2_est  = zeros(n2,2,M);

% preallocate performance measures
R2     = zeros(3,2,M);
R2_adj = zeros(3,2,M);
r      = zeros(3,2,M);
r_SC   = zeros(3,2,M);
MAE    = zeros(3,2,M);
RMSE   = zeros(3,2,M);
Obj2   = zeros(3,2,M);

% for all models
for j = 1:M
    
    % training and validation accuracy
    %---------------------------------------------------------------------%
    
    % get training and validation data
    y1; X1; y2; X2;
    
    %%% GLM = general linear model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if strcmp(mods{j},'GLM')
        b1_est = (X1'*X1)^(-1) * X1'*y1;
        y1_est(:,1,j) = X1 * b1_est;
        y2_est(:,1,j) = X2 * b1_est;
    end;
    
    %%% SVR = support vector regression %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if strcmp(mods{j},'SVR')
        svm1 = fitrsvm(X1, y1);
        y1_est(:,1,j) = predict(svm1, X1);
        y2_est(:,1,j) = predict(svm1, X2);
      % svm1 = svmtrain(y1, X1, '-s 3 -t 0 -q');
      % y1_est(:,1,j) = svmpredict(y1, X1, svm1, '-q');
      % y2_est(:,1,j) = svmpredict(y2, X2, svm1, '-q');
    end;
    
    %%% DNN = deep neural networks %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if strcmp(mods{j},'DNN')
        [X1d, X2d] = ME_prep_deep(X1, X2);
        dnn1 = trainNetwork(X1d, y1, layers, options);
        y1_est(:,1,j) = predict(dnn1, X1d);
        y2_est(:,1,j) = predict(dnn1, X2d);
    end;
    
    % transform distribution
    y1_est(:,2,j) = MD_trans_dist(y1_est(:,1,j), y1);
    y2_est(:,2,j) = MD_trans_dist(y2_est(:,1,j), y1);
    
    % tuning accuracy
    %---------------------------------------------------------------------%
    
    % for all CV folds
    for h = 1:S
        
        % get training and tuning data
        y1a = y1(i1a{h});
        X1a = X1(i1a{h},:);
        y1b = y1(i1b{h});
        X1b = X1(i1b{h},:);
        
        %%% GLM = general linear model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if strcmp(mods{j},'GLM')
            b1a_est = (X1a'*X1a)^(-1) * X1a'*y1a;
            y1b_est(i1b{h},1,j) = X1b * b1a_est;
        end;
        
        %%% SVR = support vector regression %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if strcmp(mods{j},'SVR')
            svm1a = fitrsvm(X1a, y1a);
            y1b_est(i1b{h},1,j) = predict(svm1a, X1b);
          % svm1a = svmtrain(y1a, X1a, '-s 3 -t 0 -q');
          % y1b_est(i1b{h},1,j) = svmpredict(y1b, X1b, svm1a, '-q');
        end;
        
        %%% DNN = deep neural networks %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if strcmp(mods{j},'DNN')
            [X1ad, X1bd] = ME_prep_deep(X1a, X1b);
            dnn1a = trainNetwork(X1ad, y1a, layers, options);
            y1b_est(i1b{h},1,j) = predict(dnn1a, X1bd);
        end;
        
        % transform distribution
        y1b_est(i1b{h},2,j) = MD_trans_dist(y1b_est(i1b{h},1,j), y1a);
        
    end;
    
    % prediction accuracies
    %---------------------------------------------------------------------%
    
    % for all methods
    for k = 1:K
        [R2(1,k,j), R2_adj(1,k,j), r(1,k,j), r_SC(1,k,j), MAE(1,k,j), RMSE(1,k,j), Obj2(1,k,j)] = ...
            ME_meas_corr(y1, y1_est(:,k,j),  p1);
        [R2(2,k,j), R2_adj(2,k,j), r(2,k,j), r_SC(2,k,j), MAE(2,k,j), RMSE(2,k,j), Obj2(2,k,j)] = ...
            ME_meas_corr(y1, y1b_est(:,k,j), p1);
        [R2(3,k,j), R2_adj(3,k,j), r(3,k,j), r_SC(3,k,j), MAE(3,k,j), RMSE(3,k,j), Obj2(3,k,j)] = ...
            ME_meas_corr(y2, y2_est(:,k,j),  p2);
    end;
    
end;

% prepare histograms
n_bin = numel(y_bin);
y_min = min(y_bin)-(dy/2);
y_max = max(y_bin)+(dy/2);

% preallocate histograms
n1_est  = zeros(2,n_bin,M);
n1b_est = zeros(2,n_bin,M);
n2_est  = zeros(2,n_bin,M);

% generate histograms
n1_obs = hist(y1, y_bin);
n2_obs = hist(y2, y_bin);
for j = 1:M
    for k = 1:K
        n1_est(k,:,j)  = hist(y1_est(:,k,j),  y_bin);
        n1b_est(k,:,j) = hist(y1b_est(:,k,j), y_bin);
        n2_est(k,:,j)  = hist(y2_est(:,k,j),  y_bin);
    end;
end;


%%% Step 2: Display results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% save results
save('PAC_estimate.mat', 'mods', 'meth', 'meas', 'S', 'dy', 'y_bin', 'layers', 'options', ...
                         'y1', 'X1', 'y2', 'X2', 'i1a', 'i1b', 'y1_est', 'y1b_est', 'y2_est', ...
                         'n1_obs', 'n2_obs', 'n1_est', 'n1b_est', 'n2_est', ...
                         'R2', 'R2_adj', 'r', 'r_SC', 'MAE', 'RMSE', 'Obj2');

% display models
figure('Name', 'PAC 2019: models', 'Color', [1 1 1], 'Position', [50 50 1600 900]);

subplot(2,10,1);
imagesc(y1);
caxis([y_min, y_max]);
set(gca,'XTick',[]);
ylabel('subject', 'FontSize', 16);
title('y_1 (obs.)', 'FontSize', 20);

subplot(2,10,[2:10]);
imagesc(X1);
caxis([-1, +1]);
set(gca,'YTick',[]);
xlabel('regressor', 'FontSize', 16);
title('X_1', 'FontSize', 20);

subplot(2,10,11);
imagesc(y2_est(:,k2d,j2d));
caxis([y_min, y_max]);
set(gca,'XTick',[]);
ylabel('subject', 'FontSize', 16);
title(sprintf('y_2 (%s %s)', mods{j2d}, meth{k2d}), 'FontSize', 20);

subplot(2,10,[12:20]);
imagesc(X2);
caxis([-1, +1]);
set(gca,'YTick',[]);
xlabel('regressor', 'FontSize', 16);
title('X_2', 'FontSize', 20);

% display predictions
figure('Name', 'PAC 2019: predictions', 'Color', [1 1 1], 'Position', [50 50 1600 900]);

for j = 1:M
    for k = 1:K
        subplot(K,M,(k-1)*M+j); hold on;
        plot(y1, y1b_est(:,k,j), strcat('.',cols(j)), 'MarkerSize', 1);
        plot([y_min, y_max], [y_min, y_max], '-k', 'LineWidth', 1);
        axis([y_min, y_max, y_min, y_max]);
        axis square
        set(gca,'Box','On');
        xlabel('actual age', 'FontSize', 12);
        ylabel('predicted age', 'FontSize', 12);
        title(sprintf('%s %s', mods{j}, meth{k}), 'FontSize', 16);
    end;
end;

% display correlations
figure('Name', 'PAC 2019: correlations', 'Color', [1 1 1], 'Position', [50 50 1600 900]);

for l = 1:7
    subplot(1,7,l);  hold on;
    for k = 1:K
        for j = 1:M
            if l == 1, bar((k-1)*M+j, R2(2,k,j),     cols(j)); end;
            if l == 2, bar((k-1)*M+j, R2_adj(2,k,j), cols(j)); end;
            if l == 3, bar((k-1)*M+j, r(2,k,j),      cols(j)); end;
            if l == 4, bar((k-1)*M+j, r_SC(2,k,j),   cols(j)); end;
            if l == 5, bar((k-1)*M+j, MAE(2,k,j),    cols(j)); end;
            if l == 6, bar((k-1)*M+j, RMSE(2,k,j),   cols(j)); end;
            if l == 7, bar((k-1)*M+j, Obj2(2,k,j),   cols(j)); end;
        end;
    end;
    axis([(1-0.5), (M*K+0.5), 0, 1]);
    if l == 5, ylim([0, (11/10)*max(reshape(MAE(2,:,:),1,[]))]); end;
    if l == 6, ylim([0, (11/10)*max(reshape(RMSE(2,:,:),1,[]))]); end;
    set(gca,'Box','On');
    set(gca,'XTick',[1:M*K]);
    if l == 1, legend(mods, 'Location', 'North'); end;
    xlabel('Analysis', 'FontSize', 12);
    if l == 1, ylabel('Performance', 'FontSize', 12); end;
    title(meas{l}, 'FontSize', 16);
end;

% display distributions
figure('Name', 'PAC 2019: distributions (1)', 'Color', [1 1 1], 'Position', [50 50 1600 900]);

for j = 1:M
    for k = 1:K
        subplot(K,M,(k-1)*M+j);
        bar(y_bin, n1b_est(k,:,j), cols(j));
        axis([y_min, y_max, 0, (11/10)*max(n1b_est(k,:,j))]);
        xlabel('predicted age', 'FontSize', 12);
        ylabel('number of subjects', 'FontSize', 12);
        title(sprintf('%s %s', mods{j}, meth{k}), 'FontSize', 16);
    end;
end;

figure('Name', 'PAC 2019: distributions (2)', 'Color', [1 1 1], 'Position', [50 50 1600 900]);

for j = 1:M
    % training data
    subplot(2,M,0+j); hold on;
    plot(y1b_est(:,1,j), y1b_est(:,2,j), strcat('o',cols(j)), 'MarkerSize', 1);
    plot([y_min, y_max], [y_min, y_max], '-k', 'LineWidth', 1);
    axis([y_min, y_max, y_min, y_max]);
    axis square;
    set(gca,'Box','On');
    xlabel(sprintf('predicted age (%s)', meth{1}), 'FontSize', 12);
    ylabel(sprintf('predicted age (%s)', meth{2}), 'FontSize', 12);
    title(sprintf('Training Data: %s', mods{j}), 'FontSize', 16);
    % validation data
    subplot(2,M,M+j); hold on;
    plot(y2_est(:,1,j), y2_est(:,2,j), strcat('o',cols(j)), 'MarkerSize', 1);
    plot([y_min, y_max], [y_min, y_max], '-k', 'LineWidth', 1);
    axis([y_min, y_max, y_min, y_max]);
    axis square;
    set(gca,'Box','On');
    xlabel(sprintf('predicted age (%s)', meth{1}), 'FontSize', 12);
    ylabel(sprintf('predicted age (%s)', meth{2}), 'FontSize', 12);
    title(sprintf('Validation Data: %s', mods{j}), 'FontSize', 16);
end;

figure('Name', 'PAC 2019: distributions (3)', 'Color', [1 1 1], 'Position', [50 50 1600 900]);

% training data
subplot(1,2,1);
bar(y_bin, n1_obs, 'c');
axis([y_min, y_max, 0, (11/10)*max(n1_obs)]);
xlabel('actual age', 'FontSize', 12);
ylabel('number of subjects', 'FontSize', 12);
title('Training Data', 'FontSize', 16);

% validation data
subplot(1,2,2);
bar(y_bin, n2_obs, 'm');
axis([y_min, y_max, 0, (11/10)*max(n2_obs)]);
xlabel('actual age', 'FontSize', 12);
ylabel('number of subjects', 'FontSize', 12);
title('Validation Data', 'FontSize', 16);