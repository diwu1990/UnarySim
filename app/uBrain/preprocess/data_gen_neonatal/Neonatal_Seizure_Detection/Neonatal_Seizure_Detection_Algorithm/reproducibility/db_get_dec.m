function bb=db_get_dec(path,dur_min,z0)
fn = [path,'eeg' num2str(z0) '.edf'];
[data_mont, sc, fs] = read_data_montage(fn);    % Load EEG data
anew1 = zeros(length(data_mont), length(data_mont{1})/fs(1));
anew2 = anew1;
% DB algorithm
for ii = 1:18
    dat = preprocess(double(data_mont{ii}).*sc); % Preprocessing
    anew1(ii,:) = oscillatory_type(dat, fs(ii));          % Detect oscillatory type activity
    anew2(ii,:) = spike_detection_deburch(dat, fs(ii));   % Detect of spike trains
end
aa = sum([anew1 ; anew2]);  % combine detections (any detection in either detector is a detection)
aa(aa>1)=1; 
bb = check_s_len(aa, dur_min); % remove detections less than 8s in length
disp(z0)