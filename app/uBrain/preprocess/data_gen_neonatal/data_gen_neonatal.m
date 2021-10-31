%% generate csv from neonatal eeg data
clear, clear all;
%% add path of libs
addpath(genpath("E:\ubrain_local\Neonatal_Seizure_Detection\Neonatal_Seizure_Detection_Algorithm"));
%% add path of data source
dirDataRaw = "E:\ubrain_local\neonatal_eeg";
dirDataOut = "E:\ubrain_local\neonatal_eeg_out";
%% Load all annotation data
pathAnnotation = fullfile(dirDataRaw,"annotations_2017.mat");
load(pathAnnotation);
%% Iterate over path
numDataFile = 79;
thresAnnotate = 1;
numChan = 19;
for cntDataFile = 1:numDataFile
    nameDataFileCurr = "eeg" + string(cntDataFile) + ".edf";
    nameDataOutCurr = "eeg" + string(cntDataFile) + ".csv";
    nameLabelCurr = "eeg" + string(cntDataFile) + ".label.csv";
    pathDataFileCurr = fullfile(dirDataRaw,nameDataFileCurr);
    pathDataOutCurr = fullfile(dirDataOut,nameDataOutCurr);
    pathLabelCurr = fullfile(dirDataOut,nameLabelCurr);
    %[data_mont, label, sc, fs] = read_data_montage(pathDataFileCurr);
    [dat, hdr, label_chan, fs, scle, offs] = read_edf(pathDataFileCurr);
    dataMat = zeros(length(dat{1}), numChan);
    label_header = cell(1, numChan);
    for cntChan = 1:numChan
        dataMat(:, cntChan) = dat{cntChan};
    end
    % get channel name label
    prefix = 'EEG ';  % The prefix to remove
    suffix = '-REF';  % The suffix to remove
    regex  = {['^' prefix],[suffix '$']}; % the regular expressions for prefix & suffix
    replacements = {'',''}; % Replacement strings
    for cntChan = 1:numChan
        label_header{cntChan} = convertStringsToChars(regexprep(strtrim(convertCharsToStrings(label_chan{cntChan})),...
                                regex, replacements));
    end
    
    dataTable = array2table(dataMat);
    % add channel name
    dataTable.Properties.VariableNames(1:numChan) = label_header;
    writetable(dataTable,pathDataOutCurr);
    %plot(1:length(dataMat(:,2)), dataMat(:,2))
    % get current annnotation
    annotateCurr = annotat_new{cntDataFile};
    % merge rows by taking the union
    annotateMergeCurr = sum(annotateCurr,1);
    annotateMergeCurr(annotateMergeCurr >= thresAnnotate) = 1;
    annotateMergeCurr(annotateMergeCurr < thresAnnotate) = 0;
    
    %% intepolate annotation
    scaleOri = 1:1:length(annotateMergeCurr);
    scaleTar = 1:1:length(dataMat(:,1));
    annotateInteCurr = interp1(1:length(annotateMergeCurr),...
        annotateMergeCurr,linspace(1,length(annotateMergeCurr),length(dataMat(:,1))));
    annotateTabel = array2table(annotateInteCurr');
    annotateTabel.Properties.VariableNames(1) = {'labels'};
    writetable(annotateTabel,pathLabelCurr);

end
