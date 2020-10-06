# PAC_2019

**Data analysis for Predictive Analytics Competition 2019 and Frontiers Special Issue 2020**

This code belongs to the preprint "Distributional Transformation improves Decoding Accuracy when Predicting Chronological Age from Structural MRI", submitted to the special issue "[Predicting Chronological Age from Structural Neuroimaging: The Predictive Analytics Competition 2019](https://www.frontiersin.org/research-topics/13501/predicting-chronological-age-from-structural-neuroimaging-the-predictive-analytics-competition-2019)" of *Frontiers in Psychiatry* (Section: *Computational Psychiatry*). It consists of MATLAB scripts for feature extraction (Section 2.2), decoding analyses (Section 2.3) and results display (Section 3).

- Preprint: https://www.biorxiv.org/content/10.1101/2020.09.11.293811v1
- Code: https://github.com/JoramSoch/PAC_2019

### Requirements

This code was developed and run using the following software:
- Windows 10 Pro 64-bit
- [MATLAB R2019b](https://de.mathworks.com/help/matlab/release-notes.html) (Version 9.6)
- [MATLAB's Statistics/ML Toolbox](https://de.mathworks.com/help/stats/index.html)
- [MATLAB's Deep Learning Toolbox](https://de.mathworks.com/help/deeplearning/index.html)
- [SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) (Revision 7771 as of 13/01/2020)

### Scripts

This repository contains the following analysis scripts:
- `PAC_specify.m`: allows to extract gray matter and white matter densities from structural MRI data as well as subject covariates from CSV files (Section 2.2); results are saved into `PAC_specify.mat`. Note that this script only works, if you place pre-processed data supplied within the competition into the appropriate directories on your computer (see [ll. 14ff.](https://github.com/JoramSoch/PAC_2019/blob/master/PAC_specify.m#L14) of this script).
- `PAC_specify_test_age.m`: saves age values of the validation set subjects which were not known during the competition, but only released afterwards (Section 2.1); results are saved into `PAC_specify_test_age.mat`.
- `PAC_estimate.m`: loads feature extraction results and runs the decoding analyses described in the paper (Section 2.3), i.e. multiple linear regression, support vector regression and deep neural network regression; results are saved into `PAC_estimate.mat`.
- `PAC_display/Figures.m`: loads decoding analysis results and creates figures as displayed in the paper; results were saved into `PAC_display/Figures.pdf`.

### Functions

This repository contains the following auxiliary functions:
- `ME_prep_deep.m`: prepares features for deep learning (Section 2.3.3)
- `MD_trans_dist.m`: performs distributional transformation (Section 2.4)
- `ME_meas_corr.m`: calculates performance measures (Section 2.5)
- `PAC_display/MD_pmf.m`: estimates [probability mass function](https://statproofbook.github.io/D/pmf)
- `PAC_display/MD_KL.m`: estimates [Kullback-Leibler divergence](https://statproofbook.github.io/D/kl)