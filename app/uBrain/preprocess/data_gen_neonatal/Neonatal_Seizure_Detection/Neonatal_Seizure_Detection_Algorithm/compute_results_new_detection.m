function [auc, tdr1, fdr1] = compute_results_new_detection(dec_raw, annotat, detector)
% This function estimates performance measures comparing the SVM
% output in variable dec_raw, with the annotation of a single human expert in
% variable annotat for a single recording. Variable cn is the optimal length of the collaring
% extension
%
%  auc - is the auc 
%  acc1 - is the accuracy over a range of 1000 thresholds per baby 1000x79
%  tdr1 - the seizure detection rate with 1000 thresholds
%  fdr1 - the false detection rate per hour with 1000 thresholds

%  Data must be sample with a 4s update in this case
%

% CHANGE THE SVM OUTPUT INTO A BINARY DECISION
% Steps are mean filter of 3 samples (12s) per channel
% then maximum across channels
% then a median filter of 3 samples (12s)
addpath(genpath('neonatal_sez_det'))

% SDA
if isequal(detector,'SDA')
    cn=23;
% SDA_T
elseif isequal(detector,'SDA_T')
    cn=7;
% SDA_DB_mod
elseif isequal(detector,'SDA_DB_mod')
    cn=28;    
end

M = size(dec_raw);

dum = zeros(length(dec_raw), length(dec_raw{1})); % Post processing
for jj = 1:length(dec_raw)
    dd = conv(dec_raw{jj}, ones(1,3))/3;
    dum(jj, :) = dd(2:end-1);
end
d1 = max(dum);                  % Post processing
d1 = medfilt1(d1,3);           % Post processing
d2 = [];
for kk = 1:length(d1)          % convert to 4 second samples to 1 second sample
    d2 = [d2 d1(kk).*ones(1,4)];
end
d4 = d2;
val = [min(d2) max(d2)] ;% get range of outputs        

% estimate performance over a range of thresholds per baby
% values include sensitivity, specificty, seizure detection rate and false
% alarms per hour
th = linspace(min(min(val)), max(max(val)), 1000); % changed from 1000
acc1 = zeros(length(th),1); fdr1 = acc1; tdr1 = fdr1;  spec = acc1; sens = acc1;

for z1 = 1:length(th)
    d2 = zeros(1,length(d4));  
    a = annotat;
    d2(d4>th(z1)) = 1; 
    d2 = check_s_len(d2, 10);       % eliminate detections less than 10s  
    r1 = find(diff([0 d2 0]) == 1);  % this is the collaring process (extend detection for cn seconds)
    r2 = find(diff([0 d2 0]) == -1);
    r2 = r2+cn; r2(r2>length(d2)) = length(d2);
    for z3 = 1:length(r2)
        d2(r1(z3):r2(z3)) = 1;
    end
    if length(a)>length(d2);  a = a(1:length(d2)); end
    if length(d2)>length(a);  d2 = d2(1:length(a)); end
    d2 = d2(isnan(d4)==0);
    a = a(1:length(d4));
    a = a(isnan(d4)==0);
    
    acc1(z1) = (length(find(d2==1 & a == 1))  + length(find(d2==0 & a== 0))) / sum(a == 1 | a==0);
    % for seizure babies only compare consensus annotations for sensitivity
    % and seizure detection rate
    if sum(a)>0
        [td, ~, ~, ~, ~, ~, sez_events, ~, ~, sens(z1), ~, ~, ~] = analyse_results_v2(a, d2);
        tdr1(z1) = td./sez_events;
    else
        tdr1(z1) = NaN; % Nan means baby could not be assessed as they had no seizures
    end
    % for every baby evaluate specificity and false detection rate
    [~, ~, ~, ~, fd, ~, ~, ~, ~, ~, spec(z1), ~, ~] = analyse_results_v2(a, d2);
    N = length(find(a==0)) + length(find(a==1));
    fdr1(z1) = fd/(N/3600);      
end

if isnan(sum(tdr1)) == 0 
    auc = polyarea([0 spec' 1 0], [1 sens' 0 0]);
else
    auc = NaN; % cannot estimate AUC no seizure
end

