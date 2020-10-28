% PAC 2019: visualize results (for Frontiers 2020 Special Issue)
% _
% Predictive Analytics Competition 2019: results display
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Date  : 28/10/2020, 20:54


clear
close all

%%% Step 0: Load everything %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% save results
% save('PAC_estimate.mat', 'mods', 'meth', 'meas', 'S', 'dy', 'y_bin', 'layers', 'options', ...
%                          'y1', 'X1', 'y2', 'X2', 'i1a', 'i1b', 'y1_est', 'y1b_est', 'y2_est', ...
%                          'n1_obs', 'n2_obs', 'n1_est', 'n1b_est', 'n2_est', ...
%                          'R2', 'R2_adj', 'r', 'r_SC', 'MAE', 'RMSE', 'Obj2');

% load results
PAC_dir = '../';
load(strcat(PAC_dir,'PAC_estimate.mat'));

% analysis dimensions
cols = ['grb'];
meth = {'without DT', 'with DT'};
M = numel(mods);
K = numel(meth);

% data (age) bounds
y_mm = [15,  90;
         0, 100];

% figure position
fo = [1280, -120];
fs = [1600, 900]; % *(5/4);

% specify histograms
dy    = 2.5;
y_bin = [(y_mm(2,1)+(dy/2)):dy:(y_mm(2,2)-(dy/2))];

% specify confidence intervals
alpha = [0.1, 0.05, 0.001, 0.05/250];


%%% Step 1: Process data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Figure 1: get dimensions
     p0  = 116;
[n1, p1] = size(X1);
[n2, p2] = size(X2);

% Figure 2: fit regression line
y2_mn = zeros(2,K,M);
for j = 1:M
    for k = 1:K
        y2_mn(:,k,j) = polyfit(y2, y2_est(:,k,j), 1)';
    end;
end;

% Table 4: Wilcoxon signed-rank test
y2_AE = abs(repmat(y2,[1, K, M]) - y2_est);
% y2_AE = zeros(size(y2_est));
% for j = 1:M
%     for k = 1:K
%         y2_AE(:,k,j) = abs(y2-y2_est(:,k,j));
%     end;
% end;
y2_p1 = zeros(K,M);
y2_p2 = zeros(1,M);
y2_z1 = zeros(K,M);
y2_z2 = zeros(1,M);
for j = 1:M
    for k = 1:K
       [y2_p1(k,j), h, stats] = signrank(y2_AE(:,k,1), y2_AE(:,k,j));
        if j ~= 1, y2_z1(k,j) = stats.zval; end;
    end;
   [y2_p2(j), h, stats] = signrank(y2_AE(:,2,j), y2_AE(:,1,j));
    y2_z2(j) = stats.zval;
end;

% Figure 4: generate histograms
n_bin  = numel(y_bin);
f1_obs = MD_pmf(y1, y_bin);
f2_obs = MD_pmf(y2, y_bin);
f21_KL = MD_KL(f2_obs, f1_obs);
f22_KL = MD_KL(f2_obs, f2_obs);
f2_est = zeros(K, n_bin, M);
f2e_KL = zeros(K, M);
for j = 1:M
    for k = 1:K
        f2_est(k,:,j) = MD_pmf(y2_est(:,k,j), y_bin);
        f2e_KL(k,j)   = MD_KL(f2_obs, f2_est(k,:,j));
    end;
end;

% Figure 4: Kolmogorov–Smirnov test
[h, y21_p, y21_D] = kstest2(y2, y1);
[h, y22_p, y22_D] = kstest2(y2, y2);
y2e_D = zeros(K, M);
y2e_p = zeros(K, M);
for j = 1:M
    for k = 1:K
        [h, y2e_p(k,j), y2e_D(k,j)] = kstest2(y2, y2_est(:,k,j));
    end;
end;

% Figure 5: confidence intervals
y = [y1; y2];
X = [X1; X2];
n = size(X,1);
p = size(X,2);
C = eye(p);
covB   = (X'*X)^(-1);
b_est  = (X'*X)^(-1) * X'*y;
s2_est = 1/(n-p) * (y-X*b_est)'*(y-X*b_est);
cb  = C'*b_est;
SE  = diag(sqrt(s2_est * C'*covB*C));
CI  = zeros(numel(cb),numel(alpha));
sig = false(numel(cb),numel(alpha));
for i = 1:numel(alpha)
    CI(:,i)  = 2*SE*norminv(1-alpha(i)/2, 0, 1);
    sig(:,i) = sign(cb-CI(:,i)/2)==sign(cb+CI(:,i)/2);
end;


%%% Step 2: Display results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Figure 1: data and design
figure('Name', 'PAC 2019: Figure 1', 'Color', [1 1 1], 'Position', [fo, fs]);

subplot(2,10,1);
imagesc(y1);
caxis([y_mm(1,:)]);
axis([(1-0.5), (1+0.5), (1-0.5), (n1+0.5)]);
axis ij;
colorbar;
set(gca,'Box','On');
set(gca,'XTick',[1],'XTickLabel',{'[yrs]'});
xlabel('age', 'FontSize', 12);
ylabel('subject', 'FontSize', 12);
title('y_1 [training data]', 'FontSize', 16);

subplot(2,10,[2:10]); hold on;
imagesc(X1);
plot(p0*[1 1]+0.5, [(1-0.5), (n1+0.5)], '-k');
plot(p0*[2 2]+0.5, [(1-0.5), (n1+0.5)], '-k');
caxis([-1, +1]);
axis([(1-0.5), (p1+0.5), (1-0.5), (n1+0.5)]);
axis ij;
colorbar;
set(gca,'Box','On');
set(gca,'YTick',[]);
set(gca,'XTick',[(1/2)*p0, (3/2)*p0, 2*p0+8, p1],'XTickLabel',{'GM', 'WM', 'site', 'sex'});
xlabel('regressor', 'FontSize', 12);
title(sprintf('X_1 [%d x %d training design matrix]', n1, p1), 'FontSize', 16);

subplot(2,10,10+1);
imagesc(y2);
caxis([y_mm(1,:)]);
axis([(1-0.5), (1+0.5), (1-0.5), (n2+0.5)]);
axis ij;
colorbar;
set(gca,'Box','On');
set(gca,'XTick',[1],'XTickLabel',{'[yrs]'});
xlabel('age', 'FontSize', 12);
ylabel('subject', 'FontSize', 12);
title('y_2 [validation data]', 'FontSize', 16);

subplot(2,10,10+[2:10]); hold on;
imagesc(X2);
plot(p0*[1 1]+0.5, [(1-0.5), (n2+0.5)], '-k');
plot(p0*[2 2]+0.5, [(1-0.5), (n2+0.5)], '-k');
caxis([-1, +1]);
axis([(1-0.5), (p2+0.5), (1-0.5), (n2+0.5)]);
axis ij;
colorbar;
set(gca,'Box','On');
set(gca,'YTick',[]);
set(gca,'XTick',[(1/2)*p0, (3/2)*p0, 2*p0+8, p2],'XTickLabel',{'GM', 'WM', 'site', 'sex'});
xlabel('regressor', 'FontSize', 12);
title(sprintf('X_2 [%d x %d validation design matrix]', n2, p2), 'FontSize', 16);

% Figure 2: predicted vs. actual
figure('Name', 'PAC 2019: Figure 2', 'Color', [1 1 1], 'Position', [fo, fs]);

for j = 1:M
    for k = 1:K
        subplot(K,M,(k-1)*M+j); hold on;
        plot(y2, y2_est(:,k,j), strcat('.',cols(j)), 'MarkerSize', 5);
        plot([y_mm(1,:)], [y2_mn(1,k,j)*y_mm(1,:) + y2_mn(2,k,j)], '--k', 'LineWidth', 1);
        plot([y_mm(1,:)], [y_mm(1,:)], '-k', 'LineWidth', 1);
        axis([y_mm(1,:), y_mm(1,:)]);
        axis square;
        set(gca,'Box','On');
        xlabel('actual age', 'FontSize', 12);
        ylabel('predicted age', 'FontSize', 12);
        title(sprintf('%s %s', mods{j}, meth{k}), 'FontSize', 16);
        text(y_mm(1,1)+20, y_mm(1,2)-5, sprintf('r = %1.2f\nMAE = %1.2f', r(3,k,j), MAE(3,k,j)), 'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Top');
    end;
end;

% Table 4: prediction accuracies
fprintf('\n-> Prediction performance across models and methods:');
fprintf('\n   - prediction performance:');
for k = 1:K
    fprintf('\n     - %s:', meth{k});
    for j = 1:M
        fprintf('\n       - %s: r = %1.2f, MAE = %1.2f.', mods{j}, r(3,k,j), MAE(3,k,j));
    end;
end;
fprintf('\n   - comparison of models:');
for k = 1:K
    fprintf('\n     - %s:', meth{k});
    for j = 2:M
        if y2_p1(k,j) < 0.001
            fprintf('\n       - GLM vs. %s: z = %1.2f, p < 0.001.', mods{j}, y2_z1(k,j));
        else
            fprintf('\n       - GLM vs. %s: z = %1.2f, p = %1.3f.', mods{j}, y2_z1(k,j), y2_p1(k,j));
        end;
    end;
end;
fprintf('\n   - comparison of methods:');
fprintf('\n     - %s vs. %s:', meth{2}, meth{1});
for j = 1:M
    if y2_p2(j) < 0.001
        fprintf('\n       - %s: z = %1.2f, p < 0.001.', mods{j}, y2_z2(j));
    else
        fprintf('\n       - %s: z = %1.2f, p = %1.3f.', mods{j}, y2_z2(j), y2_p2(j));
    end;
end;
fprintf('\n\n');

% Figure 3: prediction accuracies
figure('Name', 'PAC 2019: Figure 3', 'Color', [1 1 1], 'Position', [fo, fs]);
sp = {1, 5, 2, 6, 3, 7, [4,8]};

for l = 1:numel(meas)
  % subplot(1,7,l);
    subplot(2,4,sp{l});
    hold on;
    switch l
        case 1, Acc = squeeze(R2(3,:,:));
        case 2, Acc = squeeze(R2_adj(3,:,:));
        case 3, Acc = squeeze(r(3,:,:));
        case 4, Acc = squeeze(r_SC(3,:,:));
        case 5, Acc = squeeze(MAE(3,:,:));
        case 6, Acc = squeeze(RMSE(3,:,:));
        case 7, Acc = squeeze(Obj2(3,:,:));
    end;
    h = bar(Acc, 'grouped');
    for j = 1:numel(h), set(h(j), 'FaceColor', cols(j)); end;
    axis([(1-0.5), (2+0.5), 0, 1]);
    if l == 5 || l == 6, ylim([0, (11/10)*max(max(Acc))]); end;
    set(gca,'Box','On');
    set(gca,'XTick',[1:2],'XTickLabel',meth);
    if l == 7, legend(mods, 'Location', 'North'); end;
    xlabel('Analysis', 'FontSize', 12);
    ylabel('Performance', 'FontSize', 12);
    title(meas{l}, 'FontSize', 16);
    if l == 1, title('R^2_{ }', 'FontSize', 16); end;
    if l == 6, title('RMSE_{ }', 'FontSize', 16); end;
end;

% Figure 4: empirical distributions
figure('Name', 'PAC 2019: Figure 4', 'Color', [1 1 1], 'Position', [fo, fs]);

% training data
subplot(3,numel(mods)+1,1);
bar(y_bin, f1_obs, 'y');
axis([y_mm(1,:), 0, (11/10)*max(f1_obs)]);
xlabel('actual age', 'FontSize', 12);
ylabel('relative frequency', 'FontSize', 12);
title('observed [training]', 'FontSize', 16);
text(max(y_mm(1,:)), max(f1_obs), sprintf('KL = %0.3f   \nD = %0.3f   \np = %0.3f   ', f21_KL, y21_D, y21_p), 'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Top');

% validation data
subplot(3,numel(mods)+1,(numel(mods)+1)+1);
bar(y_bin, f2_obs, 'y');
axis([y_mm(1,:), 0, (11/10)*max(f2_obs)]);
xlabel('actual age', 'FontSize', 12);
ylabel('relative frequency', 'FontSize', 12);
title('observed [validation]', 'FontSize', 16);
text(max(y_mm(1,:)), max(f2_obs), sprintf('KL = %0.3f   \nD = %0.3f   \np = %0.3f   ', f22_KL, y22_D, y22_p), 'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Top');

% model predictions
for j = 1:M
    for k = 1:K
        subplot(3,numel(mods)+1,(k-1)*(numel(mods)+1)+(j+1));
        bar(y_bin, f2_est(k,:,j), cols(j));
        axis([y_mm(2,:), 0, (11/10)*max(f2_est(k,:,j))]);
        xlabel('predicted age', 'FontSize', 12);
        ylabel('relative frequency', 'FontSize', 12);
        title(sprintf('%s %s', mods{j}, meth{k}), 'FontSize', 16);
        if y2e_p(k,j) < 0.001
            text(max(y_mm(2,:)), max(f2_est(k,:,j)), sprintf('KL = %0.3f   \nD = %0.3f   \np < 0.001   ', f2e_KL(k,j), y2e_D(k,j)), 'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Top');
        else
            text(max(y_mm(2,:)), max(f2_est(k,:,j)), sprintf('KL = %0.3f   \nD = %0.3f   \np = %0.3f   ', f2e_KL(k,j), y2e_D(k,j), y2e_p(k,j)), 'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Top');
        end;
    end;
end;

% distributional transformations
for j = 1:M
    subplot(3,numel(mods)+1,2*(numel(mods)+1)+(j+1)); hold on;
    plot(y2_est(:,1,j), y2_est(:,2,j), strcat('.',cols(j)), 'MarkerSize', 5);
    plot([y_mm(2,:)], [y_mm(2,:)], '-k', 'LineWidth', 1);
    axis([y_mm(2,:), y_mm(1,:)]);
    axis equal;
    set(gca,'Box','On');
    xlabel(sprintf('predicted age (%s)', meth{1}), 'FontSize', 12);
    ylabel(sprintf('predicted age (%s)', meth{2}), 'FontSize', 12);
    title(sprintf('%s: DT', mods{j}), 'FontSize', 16);
end;

% Figure 5: confidence intervals
figure('Name', 'PAC 2019: Figure 5', 'Color', [1 1 1], 'Position', [fo, fs]);
y_pos  = 117;          %  (21/20)*max(cb+CI(:,1)/2);
y_lims = [-125, +125]; % [(11/10)*min(cb-CI(:,1)/2), (11/10)*max(cb+CI(:,1)/2)]; % 

hold on;
bar([1:p], cb, 'g');
errorbar([1:p], cb, CI(:,1)/2, CI(:,1)/2, '.k', 'LineWidth', 1, 'CapSize', 1);
plot(p0*[1 1]+0.5, y_lims, '-k');
plot(p0*[2 2]+0.5, y_lims, '-k');
axis([(1-0.5), (p+0.5), y_lims]);
set(gca,'Box','On');
set(gca,'XTick',[(1/2)*p0, (3/2)*p0, 2*p0+8, p1],'XTickLabel',{'GM', 'WM', 'site', 'sex'});
xlabel('regressor', 'FontSize', 12);
ylabel(sprintf('parameter estimate and %d%% confidence interval', round((1-alpha(1))*100)), 'FontSize', 12);
title('GLM: regression coefficients', 'FontSize', 16);
for i = 1:numel(cb)
    if sig(i,2)
        text(i, y_pos+0, '*', 'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Middle');
    end;
    if sig(i,3)
        text(i, y_pos-2, '*', 'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Middle');
    end;
    if sig(i,4)
        text(i, y_pos-4, '*', 'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Middle');
    end;
end;