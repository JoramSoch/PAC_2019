% PAC 2019: specify model (for Frontiers 2020 re-analysis)
% _
% Predictive Analytics Competition 2019: model specification
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Date  : 04/08/2020, 14:03


clear

%%% Step 0: Specify parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set directories
train_dir = 'I:\joram\NBD\DataSets\Cole_et_al_2017\data\train\';
test_dir  = 'I:\joram\NBD\DataSets\Cole_et_al_2017\data\test\';
tool_dir  = 'C:\Joram\ownCloud\BCCN\NBD\DataSets\Cole_et_al_2017\tools\';
data_dir  = 'C:\Joram\ownCloud\BCCN\NBD\DataSets\Cole_et_al_2017\data\';

% set covariates
train_cov = 'PAC2019_BrainAge_Training.csv';
test_cov  = 'PAC2019_BrainAge_Test.csv';


%%% Step 1: Load AAL atlas %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% read mask image
mask_str = strcat(train_dir,'gm/sub0_gm.nii');
mask_hdr = spm_vol(mask_str);
mask_img = spm_read_vols(mask_hdr);
mask_ind = find(mask_img~=0)';

% read resliced AAL
aal_str = strcat(tool_dir,'AAL_reslice/rAAL.nii');
aal_hdr = spm_vol(aal_str);
aal_img = spm_read_vols(aal_hdr);
aal_img = aal_img(mask_ind);

% generate AAL indices
num_reg = max(aal_img(:));
aal_ind = cell(num_reg,1);
for j = 1:num_reg
    aal_ind{j} = find(aal_img==j);
end;


%%% Step 2: Load training data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get subject IDs
files     = dir(strcat(train_dir,'gm/*_gm.nii.gz'));
num_subj1 = numel(files);
subj_ids1 = cell(num_subj1,1);
for i = 1:num_subj1
    subj_ids1{i} = files(i).name(1:strfind(files(i).name,'_gm')-1);
end;
d = floor(num_subj1/100);

% read GM/WM images
spm_progress_bar('Init', 100, 'load training data: read GM/WM images', '');
GM1 = zeros(num_subj1,num_reg);
WM1 = zeros(num_subj1,num_reg);
for i = 1:num_subj1
    % gray matter
    gm_str = strcat(train_dir,'gm/',subj_ids1{i},'_gm.nii.gz');
    gm_hdr = spm_vol(gm_str);
    gm_img = spm_read_vols(gm_hdr);
    gm_img = gm_img(mask_ind);
    % white matter
    wm_str = strcat(train_dir,'wm/',subj_ids1{i},'_wm.nii.gz');
    wm_hdr = spm_vol(wm_str);
    wm_img = spm_read_vols(wm_hdr);
    wm_img = wm_img(mask_ind);
    % all regions
    for j = 1:num_reg
        GM1(i,j) = mean(gm_img(aal_ind{j}));
        WM1(i,j) = mean(wm_img(aal_ind{j}));
    end;
    if mod(i,d) == 0, spm_progress_bar('Set',(i/num_subj1)*100); end;
end;
clear gm_* wm_*
spm_progress_bar('Clear');

% read covariates
[sID, age, gender, site] = textread(strcat(data_dir,train_cov), '%s%f%s%d', 'delimiter', ',', 'headerlines', 1);
num_sites = max(site)-min(site)+1;
c1 = zeros(num_subj1,num_sites+1);
y1 = zeros(num_subj1,1);
for i = 1:num_subj1
    subj_ind = find(strcmp(sID,subj_ids1{i}));
    y1(i,1)  = age(subj_ind);
    c1(i,site(subj_ind)+1) = 1;
    if strcmp(gender{subj_ind},'m'), c1(i,end) = +1; end;
    if strcmp(gender{subj_ind},'f'), c1(i,end) = -1; end;
end;
clear sID age gender site subj_ind


%%% Step 3: Load test data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get subject IDs
files     = dir(strcat(test_dir,'gm/*_gm.nii.gz'));
num_subj2 = numel(files);
subj_ids2 = cell(num_subj2,1);
for i = 1:num_subj2
    subj_ids2{i} = files(i).name(1:strfind(files(i).name,'_gm')-1);
end;
d = floor(num_subj2/100);

% read GM/WM images
spm_progress_bar('Init', 100, 'load test data: read GM/WM images', '');
GM2 = zeros(num_subj2,num_reg);
WM2 = zeros(num_subj2,num_reg);
for i = 1:num_subj2
    % gray matter
    gm_str = strcat(test_dir,'gm/',subj_ids2{i},'_gm.nii.gz');
    gm_hdr = spm_vol(gm_str);
    gm_img = spm_read_vols(gm_hdr);
    gm_img = gm_img(mask_ind);
    % white matter
    wm_str = strcat(test_dir,'wm/',subj_ids2{i},'_wm.nii.gz');
    wm_hdr = spm_vol(wm_str);
    wm_img = spm_read_vols(wm_hdr);
    wm_img = wm_img(mask_ind);
    % all regions
    for j = 1:num_reg
        GM2(i,j) = mean(gm_img(aal_ind{j}));
        WM2(i,j) = mean(wm_img(aal_ind{j}));
    end;
    if mod(i,d) == 0, spm_progress_bar('Set',(i/num_subj2)*100); end;
end;
clear gm_* wm_*
spm_progress_bar('Clear');

% read covariates
[sID, gender, site] = textread(strcat(data_dir,test_cov), '%s%s%d', 'delimiter', ',', 'headerlines', 1);
c2 = zeros(num_subj2,num_sites+1);
for i = 1:num_subj2
    subj_ind = find(strcmp(sID,subj_ids2{i}));
    c2(i,site(subj_ind)+1) = 1;
    if strcmp(gender{subj_ind},'m'), c2(i,end) = +1; end;
    if strcmp(gender{subj_ind},'f'), c2(i,end) = -1; end;
end;
clear sID gender site subj_ind


%%% Step 4: Save everything %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% save train and test data
sID1 = subj_ids1;
sID2 = subj_ids2;
save('PAC_specify.mat', 'sID1', 'sID2', 'y1', 'GM1', 'WM1', 'c1', 'GM2', 'WM2', 'c2');