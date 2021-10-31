%% Seizure detection funtion
function [dec, dec_raw, feat] = seizure_detection(filename, format, detector, n, varargin)

% A function that implements several seizure detection algorithms on the EEG
% 
% Run the algorithm
% [dec, dec_raw, feat] = seizure_detection(filename, format, detector);

% INPUTS:

% filename is either:
%   1) the filename of an 18 channel bipolar montage (identical to that used in the EEG database)

%   2) the filename of an EDF file containing only EEG channels (arbitrary montage and channel number)

%   3) a matlab variable containing only EEG data (channel x time).

% format
% is a scalar denoting the type of data input 1, 2, 3 corresponding to the above possible filenames
% if 3 is selected then an additional sampling frequency parameter is required

% detector 
% is a string containing 4 potential values - SDA, SDA_DB_mod, SDA_T, SDA_DB

% n
% is a scalar denoting the number of cores to run the algorithm on. If multiple cores are not available, define n=1

% OPTIONAL INPUTS:

% model_file_path
% MUST BE DEFINED if detector is: SDA, SDA_DB_mod or SDA_T
% is the path to a matlab variable including 1) the model file used to implement the SVM, 
% 2) the normalization values for the features and 3) the threshold for the decision

% OUTPUTS:

% dec is the final decision/binary output annotation sampled at 1Hz

% dec_raw is the raw SVM output for each EEG channel (for SDA, SDA_DB_mod, SDA_T only)
% feat is the feature values (for SDA, SDA_DB_mod, SDA_T only)

% EXAMPLES OF USE:

% When filename is class 1: 
% filename='/path_to/eeg1.edf'
% model_file='/neonatal_sez_det/fullSVMs/fullSVM_SDA.mat';
% [dec, dec_raw, feat] = seizure_detection(filename, 1, 'SDA',  model_file);

% When filename is class 2:
% filename='/path_to/eeg_data_file1001.edf';
% [dec, dec_raw, feat] = seizure_detection(filename, 2, 'SDA_T');

% When filename is class 3:
% filename='/path_to/eeg.mat';
% dec = seizure_detection(filename, 3, 'SDA_DB', [], fs);
% Define inputs here:
% required input variables:
%   filename
%   format
%   detector
% optional input variables:
% %   model_file_path
%   fs
%
% Karoliina Tapani and Nathan Stevenson
% Aalto University and University of Helsinki, Finland
% June 2018
if format==3 && isempty(varargin)
    disp('Input variable "fs" missing')
    dec=[];
else
    addpath(genpath('neonatal_sez_det'))

    if isequal(detector,'SDA_DB')
        if n>1
            disp('No parallelization needed with SDA_DB')
            disp('Continuing with single core')
        end
        if format==3 && length(varargin)==1
            disp('Starting to compute binary annotation with SDA_DB-algorithm')
            dec=DB_algorithm_original(filename,format,varargin{1});
            disp('Binary decision done')
        else
            disp('Starting to compute binary annotation with SDA_DB-algorithm')
            dec=DB_algorithm_original(filename,format);
            disp('Binary decision done')
        end
    elseif isequal(detector,'SDA') || isequal(detector,'SDA_T') || isequal(detector,'SDA_DB_mod')
        % compute features
        load('notch_filter');
        load('hp');
        if format==3 && length(varargin)==1
            feat=compute_features(n,format,detector,filename,hp,Num,Den,varargin{1});
        elseif format==1 || format==2
            feat=compute_features(n,format,detector,filename,hp,Num,Den);
        end
        % compute binary annotation (dec) and raw decision values (dec_raw)
        load(['fullSVM_',detector])
        [dec,dec_raw]=compute_decision_values(detector,feat,mdlSVM,norm_val,thr,n);

    else
        disp('Invalid detector')
    end

end
end

    














    
