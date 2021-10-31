function [dec,results,dec_raw,feats]=reproduce_SDA(group,path,annotat,n)

%% The original seizure detection algorithm

% This is the main script of the seizure detection algorithm. The user can
% choose from four different algorithms: SDA, SDA_T, SDA_DB and SDA_DB_mod.
% With this script, features can be extracted from the whole dataset and then tested
% with the trained SVM to aquire the decision values and binary annotation for each EEG trace.
% Finally, performance measures can be estimated comparing the SVM output
% to the human annotation of three experts, that is, the gold standard.

% For more information, please refer to 
% [1] K. Tapani, S. Vanhatalo and N. Stevenson, Time-varying EEG correlations 
% improve automated neonatal seizure detection, International Journal of 
% Neural Systems. (accepted for publication)

% INPUTs
% group - The SDA implementation of choice
% path - path to the edf-files
% annotat - load the annotation file
% n -  choose the number of parallel pools (recommended to use as many as possible, if no parallelization in use, define n=1)


%% Options for group:
% Proposed SDA: group='SDA'
% SDA Temko: group='SDA_T'
% Version of the original Deburchgraeve SDA: group='SDA_DB'
% Modified version of the original Deburchgraeve SDA: group='SDA_DB_mod'

%Example of use:
%group='SDA_DB_mod';
%path='data/';
%n=15;
%[dec,results,dec_raw,feats]=reproduce_SDA(group,path,annotat,n);


%% Compute decision values, binary output and performance measures
addpath(genpath('reproducibility'))
addpath(genpath('neonatal_sez_det'))
% Original Deburchgraeve algorithm
if isequal(group,'SDA_DB')
    disp('Binary decision computed for patients:')
    dec=db_algorithm(path,n);
    
% Other algorithms
else
    fs_orig=256;
    % load filters
    load hp; load notch_filter
    feats=get_features(group,path,hp,Num,Den,fs_orig,n);
    [dec_raw,dec]=get_decision_values(group,annotat,feats,n);
end
% Compute post-processed results
if isequal(group,'SDA_DB')
    results=get_results(group,annotat,dec);
else
    results=get_results(group,annotat,dec_raw);
end



delete(gcp('nocreate'))
end
