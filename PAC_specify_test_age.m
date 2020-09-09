% PAC 2019: specify model (for Frontiers 2020 re-analysis)
% _
% Predictive Analytics Competition 2019: age of test subjects
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Date  : 13/08/2020, 14:36


clear

%%% Step 0: Specify parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set directories
train_dir = 'I:\joram\NBD\DataSets\Cole_et_al_2017\data\train\';
test_dir  = 'I:\joram\NBD\DataSets\Cole_et_al_2017\data\test\';
tool_dir  = 'C:\Joram\ownCloud\BCCN\NBD\DataSets\Cole_et_al_2017\tools\';
data_dir  = 'C:\Joram\ownCloud\BCCN\NBD\DataSets\Cole_et_al_2017\data\';

% set covariates
train_cov = 'PAC2019_BrainAge_Training.csv';
test_cov  = 'PAC2019_BrainAge_Test_age.csv';


%%% Step 2: Load test age %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% read subject IDs
load(strcat(tool_dir,'PAC_specify.mat'));
subj_ids2 = sID2;

% read covariates
[sID, age, gender, site] = textread(strcat(data_dir,test_cov), '%s%f%s%d', 'delimiter', ',', 'headerlines', 1);
n2 = numel(subj_ids2);
y2 = zeros(n2,1);
for i = 1:n2
    subj_ind = find(strcmp(sID,subj_ids2{i}));
    y2(i,1)  = age(subj_ind);
end;


%%% Step 3: Save test age %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% save train and test data
sID2 = subj_ids2;
save('PAC_specify_test_age.mat', 'sID2', 'y2');