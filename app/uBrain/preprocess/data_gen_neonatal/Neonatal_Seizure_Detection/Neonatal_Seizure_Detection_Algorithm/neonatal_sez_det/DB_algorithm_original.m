% DB algorithm
function bb=DB_algorithm_original(filename,format,varargin)

dur_min = 8; 

if format==1
    [data_mont, sc, fs] = read_data_montage(filename);    % Load EEG data
    disp('EDF-file red in')
    anew1 = zeros(length(data_mont), length(data_mont{1})/fs(1));
    anew2 = anew1;
    % DB algorithm
    disp('Computing annotation for channel number:')
    for ii = 1:length(data_mont)
        disp(ii)
        dat = preprocess(double(data_mont{ii}).*sc); % Preprocessing
        anew1(ii,:) = oscillatory_type(dat, fs(ii));          % Detect oscillatory type activity
        anew2(ii,:) = spike_detection_deburch(dat, fs(ii));   % Detect of spike trains
    end
    aa = sum([anew1 ; anew2]);  % combine detections (any detection in either detector is a detection)
    aa(aa>1)=1; 
    bb = check_s_len(aa, dur_min); % remove detections less than 8s in length

elseif format==2
    [data_mont, ~,~, fs, scle]  = read_edf(filename);
    sc = scle(1);
    anew1 = zeros(length(data_mont), length(data_mont{1})/fs(1));
    anew2 = anew1;
    disp('EDF-file red in')
    disp('Computing annotation for channel number:')
    for ii = 1:length(data_mont)
        disp(ii)
        dat = preprocess(double(data_mont{ii}).*sc); % Preprocessing
        anew1(ii,:) = oscillatory_type(dat, fs(ii));          % Detect oscillatory type activity
        anew2(ii,:) = spike_detection_deburch(dat, fs(ii));   % Detect of spike trains
    end
    aa = sum([anew1 ; anew2]);  % combine detections (any detection in either detector is a detection)
    aa(aa>1)=1; 
    bb = check_s_len(aa, dur_min); % remove detections less than 8s in length
    
elseif format==3
    fs=varargin{1};
    s=load(filename);
    x=fieldnames(s);
    eeg_data=getfield(s, x{1});
    disp('Mat-file red in')
    if iscell(eeg_data)==0
        anew1 = zeros(size(eeg_data,1), size(eeg_data,2)/fs);
        anew2 = anew1;
        rows=size(eeg_data,1);
        sig=mat2cell(eeg_data,ones(1,rows),length(eeg_data));
        disp('Computing annotation for channel number:')
        for ii = 1:size(eeg_data,1)
            disp(ii)
            dat=preprocess(sig{ii}); % Preprocessing
            anew1(ii,:) = oscillatory_type(dat, fs);          % Detect oscillatory type activity
            anew2(ii,:) = spike_detection_deburch(dat, fs);   % Detect of spike trains
        end
    else
        anew1 = zeros(length(eeg_data), length(eeg_data{1})/fs);
        anew2 = anew1;
        disp('Computing annotation for channel number:')
        for ii = 1:size(eeg_data,1)
            disp(ii)
            dat=preprocess(eeg_data{ii}); % Preprocessing
            anew1(ii,:) = oscillatory_type(dat, fs);          % Detect oscillatory type activity
            anew2(ii,:) = spike_detection_deburch(dat, fs);   % Detect of spike trains
        end
    end
    aa = sum([anew1 ; anew2]);  % combine detections (any detection in either detector is a detection)
    aa(aa>1)=1; 
    bb = check_s_len(aa, dur_min); % remove detections less than 8s in length
    
else
    disp('Format should be 1, 2 or 3')
    
end
end
