function [auc, auc1, acc1, tdr1, fdr1, tdr2, fdr2, kap, dkap, kap1, dkap_dist, thr1] = compute_results_NJS(dec_raw, annotat, cn)
% This function estimates a raft of performance measures comparing the SVM
% output in variable dec_raw, with the annotations of the human expert in
% variable annotat. Variable cn is the optimal length of the collaring
% extension
%
%  auc1 - is the auc of all recording concatenated together 1x1
%  auc - is the auc on a per baby basis 1x79 with NaN when no seizure is present in a baby
%  acc1 - is the accuracy over a range of 1000 thresholds per baby 1000x79
%  tdr1 - the seizure detection rate on a per baby basis with 1000 thresholds 1000x79
%  fdr1 - the false detection rate per hour on a baby per baby basis with 1000 thresholds 1000x79
%  tdr2 - the seizure detection rate over the concatenated recordings with 1000 thresholds 1x1000
%  fdr2 - the false detections per hour over the concantenated recordings with 1000 thresholds 1x1000
%  kap - the kappa value between human experts with a 95%CI 1x3
%  dkap - the 95%CI change in kappa when replacing a human expert with the SDA 3x2
%  kap1 - the kappa when replacing a human expert with the SDA 1x3
%  dkap_dist - differences to human expert
%
%  Data must be sample with a 4s update in this case
%

% CHANGE THE SVM OUTPUT INTO A BINARY DECISION
% Steps are mean filter of 3 samples (12s) per channel
% then maximum across channels
% then a median filter of 3 samples (12s)

M = size(dec_raw); MM = size(dec_raw{1}');
d4 = cell(1,M(1)); %acm = zeros(1,M(1)); auc = acm; %th1 = zeros(1,M(1));
val = zeros(M(1),2); 
for ii = 1:M(1)
    a = annotat{ii};
    a = sum(a);
    dum = zeros(MM(2), length(dec_raw{ii}{1})); % Post processing
    for jj = 1:MM(2)
        dd = conv(dec_raw{ii}{jj}, ones(1,3))/3;
        dum(jj, :) = dd(2:end-1);
    end
    d1 = max(dum, [], 1);          % Post processing or max(dum); ???
    d1 = medfilt1(d1,3);           % Post processing
    d2 = [];
    for kk = 1:length(d1)          % convert to 4 second samples to 1 second sample
        d2 = [d2 d1(kk).*ones(1,4)];
    end
    d4{ii} = d2;
    val(ii,:) = [min(d2) max(d2)] ;% get range of outputs        
end

% estimate performance over a range of thresholds per baby
% values include sensitivity, specificty, seizure detection rate and false
% alarms per hour
th = linspace(min(min(val)), max(max(val)), 1000); % changed from 1000
acc1 = zeros(length(th),M(1)); fdr1 = acc1; tdr1 = fdr1;  spec = acc1; sens = acc1;
for z1 = 1:length(th)
    A = []; B = []; C = [];
for ii = 1:M(1)
    a = annotat{ii};
    a = sum(a);
    d1 = d4{ii}; d2 = zeros(1,length(d1));  
    d2(d1>th(z1)) = 1; 
    d2 = check_s_len(d2, 10);       % eliminate detections less than 10s  
    r1 = find(diff([0 d2 0]) == 1);  % this is the collaring process (extend detection for cn seconds)
    r2 = find(diff([0 d2 0]) == -1);
    r2 = r2+cn; r2(r2>length(d2)) = length(d2);
    for z3 = 1:length(r2)
        d2(r1(z3):r2(z3)) = 1;
    end   
    d2 = d2(isnan(d1)==0);
    a = a(isnan(d1)==0);
    d1 = d1(isnan(d1)==0);    
    if length(a)>length(d2);  a = a(1:length(d2)); end
    if length(d2)>length(a);  d2 = d2(1:length(a)); end
    acc1(z1,ii) = (length(find(d2==1 & a == 3))  + length(find(d2==0 & a== 0))) / sum(a == 3 | a==0);
    a1 = a;
    a2 = zeros(1, length(a));
    a2(a1==3)=1;
    % for seizure babies only compare consensus annotations for sensitivity
    % and seizure detection rate
    if sum(a2)>0
        a2 = check_s_len(a2, 10);
        [td, ~, ~, ~, ~, ~, sez_events, ~, ~, sens(z1,ii), ~, ~, ~] = analyse_results_v2(a2, d2);
        tdr1(z1, ii) = td./sez_events;
    else
        tdr1(z1, ii) = NaN; % Nan means baby could not be assessed as they had no seizures
    end
    % for every baby evaluate specificity and false detection rate
    a3 = zeros(1, length(a)); 
    a3(a1>=1)=1;
    a3 = check_s_len(a3, 10);
    [~, ~, ~, ~, fd, ~, ~, ~, ~, ~, spec(z1,ii), ~, ~] = analyse_results_v2(a3, d2);
    N = length(find(a1==0)) + length(find(a1==3));
    fdr1(z1, ii) = fd/(N/3600);      
    A = [A a1];
    B = [B d2];
    C = [C d1];
end
   A1 = zeros(1, length(A));
   A1(A==3)=1;
   [td, ~, ~, ~, ~, ~, sez_events, ~, ~, ~, ~, ~, ~] = analyse_results_v2(A1, B);
   tdr2(z1) = td./sez_events;
   A2 = zeros(1, length(A)); 
   A2(A>=1)=1;
   [~, ~, ~, ~, fd, ~, ~, ~, ~, ~, ~, ~, ~] = analyse_results_v2(A2, B);
   N = length(find(A==0)) + length(find(A==3));
   fdr2(z1) = fd/(N/3600);   
end
% AUC per subject
auc = zeros(1,M(1));
for ii = 1:M(1)
    if isnan(sum(tdr1(:, ii))) == 0 
    auc(ii) = polyarea([0 spec(:,ii)' 1 0], [1 sens(:,ii)' 0 0]);
    else
    auc(ii) = NaN; % cannot estimate AUC no seizure
    end
end
% AUC on concatenated data
M1 = length(find(A==3));
M2 = length(find(A==0));
sens = zeros(1,length(th)); spec = sens;
for jj = 1:length(th)
    a1 = zeros(1,length(C));
    a1(C>th(jj)) = 1;
    a1 = check_s_len(a1, 10);
    r1 = find(diff([0 a1 0]) == 1);
    r2 = find(diff([0 a1 0]) == -1);
    r2 = r2+cn; r2(r2>length(a1)) = length(a1);
    for z3 = 1:length(r2)
       a1(r1(z3):r2(z3)) = 1;
    end
    sens(jj) = length(find(a1==1 & A==3))/M1;
    spec(jj) = length(find(a1==0 & A==0))/M2;
end
auc1 = polyarea([0 spec 1 0], [1 sens 0 0]);

%%%%%%%%%%%%%%%%%%%%%%%%%
%
% IOA 
%
% this part selects the optimal threshold for the SDA by maximising the
% agreement with the human expert.
%
%%%%%%%%%%%%%%%%%%%%%%%%

kf1 = zeros(1,length(th)); kf2 = kf1; kf3 = kf2; kf4 = kf2;
for z1 = 1:length(th)
    A = []; B = [];
for ii = 1:M(1)
    d1 = d4{ii}; d2 = zeros(1,length(d1));
    d2(d1>th(z1)) = 1; 
    d2 = check_s_len(d2, 10);
    r1 = find(diff([0 d2 0]) == 1);
    r2 = find(diff([0 d2 0]) == -1);
    r2 = r2+cn; r2(r2>length(d2)) = length(d2);
       for z3 = 1:length(r2)
           d2(r1(z3):r2(z3)) = 1;
       end     
          a = annotat{ii};
    if length(a)>length(d2);  a = a(:,1:length(d2)); end
    if length(d2)>length(a);  d2 = d2(1:length(a)); end
    d2 = d2(isnan(d1)==0);
    a = a(:,isnan(d1)==0);
    A = [A a];
    B = [B d2];
end
    AA = [A' B'];
    kf1(z1) = estimate_kf(AA);
    A2 = [A(1,:)' A(2,:)' B'];
    kf2(z1) = estimate_kf(A2);
    A3 = [A(1,:)' A(3,:)' B'];
    kf3(z1) = estimate_kf(A3);
    A4 = [A(3,:)' A(2,:)' B'];
    kf4(z1) = estimate_kf(A4);
end
kap(1) = estimate_kf(A');
ref1 = find(kf1==max(kf1), 1, 'last');
kap1 = [kf2(ref1) kf3(ref1) kf4(ref1)];
thr1 = th(ref1);
d3 = cell(1, M(1));
for ii = 1:M(1)
    d1 = d4{ii}; d2 = zeros(1,length(d1));
    d2(d1>th(ref1)) = 1; 
    d2 = check_s_len(d2, 10);
       r1 = find(diff([0 d2 0]) == 1);
       r2 = find(diff([0 d2 0]) == -1);
       r2 = r2+cn; r2(r2>length(d2)) = length(d2);
       for z3 = 1:length(r2)
           d2(r1(z3):r2(z3)) = 1;
       end     
    if length(a)>length(d2);  a = a(:,1:length(d2)); end
    if length(d2)>length(a);  d2 = d2(1:length(a)); end
    d2(isnan(d1)==1)=NaN;
    d3{ii} = d2;
end
% here is the boostrap to estimate the confidence interval of differences
% in kappa
rng(1)
ref =  round(rand(1000,M(1))*(M(1)-1))+1;
BB = size(ref); kf1 = zeros(1,1000); df1 = kf1; df2 = kf1; df3 = kf1;
for ii = 1:BB(1)
    C = []; D = []; E = []; F = []; 
    r1 = ref(ii,:);
    for jj = 1:length(r1)
        dum = d3{r1(jj)};
        drf =find(isnan(dum)==0);
        C = [C  annotat{r1(jj)}(1,drf)];
        D = [D annotat{r1(jj)}(2,drf)];
        E = [E  annotat{r1(jj)}(3,drf)];
        F = [F dum(drf)];
    end   
    A = [C' D' E'];
    kf1(ii) = estimate_kf(A);
    A = [C' D' F'];
    kf2 = estimate_kf(A);
    A = [C' F' E'];
    kf3 = estimate_kf(A);
    A = [F' D' E'];
    kf4 = estimate_kf(A);
    df1(ii) = kf1(ii)-kf2;
    df2(ii) = kf1(ii)-kf3;
    df3(ii) = kf1(ii)-kf4;  
end

kap(2:3) = quantile(kf1, [0.025 0.975]);
dkap(1,:) = quantile(df1, [0.025 0.975]); % substitute E 
dkap(2,:) = quantile(df2, [0.025 0.975]); % substitute E 
dkap(3,:) = quantile(df3, [0.025 0.975]); % substitute E 
dkap_dist(1,:)=df1;
dkap_dist(2,:)=df2;
dkap_dist(3,:)=df3;


