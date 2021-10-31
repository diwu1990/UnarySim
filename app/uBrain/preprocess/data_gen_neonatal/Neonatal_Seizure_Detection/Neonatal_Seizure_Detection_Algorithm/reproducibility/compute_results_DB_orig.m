function [auc, auc_cc,auc1, tdr1, fdr1, tdr2, fdr2, kap, dkap,ddkap,dkap_dist,sens,spec,ppv,npv] = compute_results_DB_orig(dec, annotat, cn,num)

% CHANGE THE SVM OUTPUT INTO A BINARY DECISION
% AUC per patient
MM = size(dec); 
d4= cell(1,MM(1)); % automated annotation
auc = zeros(1,MM(1)); sens=zeros(1,MM(1));spec=sens;ppv=sens;npv=sens;
for ii = 1:MM(1)
    a = annotat{ii};
    a = sum(a);  
    if length(find(a==3))>0
        xx = linspace(0,1,1000);
        M1 = length(find(a==3));
        M2 = length(find(a==0));
        % extend automated annotations with collar cn     
        d3 = dec{ii};
        r1 = find(diff([0 d3 0]) == 1);
        r2 = find(diff([0 d3 0]) == -1);
        r2 = r2+cn; r2(r2>length(d3)) = length(d3);
        for z3 = 1:length(r2)
           d3(r1(z3):r2(z3)) = 1;
        end
        d4{ii}=d3;       
        % compute statistical measures
        if isempty(find(d3==1 & a==3))
            sens(ii)=0;ppv(ii)=0;
        else
            sens(ii) = length(find(d3==1 & a==3))/M1; %sensitivity
            ppv(ii) = length(find(d3==1 & a==3))/length(find(d3==1)); %precision 
        end
        spec(ii) = length(find(d3==0 & a==0))/M2; %specificity           
        npv(ii) = length(find(d3==0 & a==0))/length(find(d3==0)); %negative predictive value   
        sn1 = pchip([0 0.5 1], [1 sens(ii) 0], xx);
        sp1 = pchip([0 0.5 1], [0 spec(ii) 1], xx);
        auc(ii) = polyarea([0 sp1 1 0], [1 sn1 0 0]);       
    end
end

% Event based assessments
fdr1 = zeros(1,MM(1)); tdr1 = fdr1; 
A = []; B = []; C = [];AA=[];
for ii = 1:MM(1)
    a = annotat{ii};
    a = sum(a);
    aa=a;
    d1 = dec{ii}; d2 = d1;
    % collar automated annotation
    r1 = find(diff([0 d2 0]) == 1);
    r2 = find(diff([0 d2 0]) == -1);
    r2 = r2+cn; r2(r2>length(d2)) = length(d2);
    for z3 = 1:length(r2)
       d2(r1(z3):r2(z3)) = 1;
    end    
    d2 = d2(isnan(d1)==0);
    a = a(isnan(d1)==0);
    d1 = d1(isnan(d1)==0);          
    if length(a)>length(d2);  a = a(1:length(d2)); end
    if length(aa)>length(d2);  aa = aa(1:length(d2)); end
    if length(d2)>length(a);  d2 = d2(1:length(a)); end
    % human annotation
    a1 = a;
    % For TDR unanimous annotation set to 1, everything else is zero
    a2 = zeros(1, length(a));
    a2(a1==3)=1;
    if sum(a2)>0
    a2 = check_s_len(a2, 8);        % eliminate detections less than 10s
    [td, ~, ~, ~, ~, ~, sez_events, ~, ~, ~, ~, ~, ~] = analyse_results_v2(a2, d2);
    tdr1(ii) = td./sez_events;
    end
    a3 = zeros(1, length(a)); 
    % For FDR all annotations >1 (at least one annotated sez) set to 1,
    % only unanimous nonseizure set to zero
    a3(a1>=1)=1;
    a3 = check_s_len(a3, 8);        % eliminate detections less than 10s
    [~, ~, ~, ~, fd, ~, ~, ~, ~, ~, ~, ~, ~] = analyse_results_v2(a3, d2);
    M = length(find(a1==0)) + length(find(a1==3));
    fdr1(ii) = fd/(M/3600);   
    A = [A a];
    AA=[AA aa]; 
    B = [B d2];
    C = [C d1];
end
A1 = zeros(1, length(A));
A1(A==3)=1;
[td, ~, ~, ~, ~, ~, sez_events, ~, ~, ~, ~, ~, ~] = analyse_results_v2(A1, B);
tdr2 = td./sez_events;
A2 = zeros(1, length(A)); 
A2(A>=1)=1;
[~, ~, ~, ~, fd, ~, ~, ~, ~, ~, ~, ~, ~] = analyse_results_v2(A2, B);
M = length(find(A==0)) + length(find(A==3));
fdr2 = fd/(M/3600);   
   
% Concatenated AUC
MM1 = length(find(AA==3));
MM2 = length(find(AA==0));
a1=C;
a1 = check_s_len(a1, 8);
r1 = find(diff([0 a1 0]) == 1);
r2 = find(diff([0 a1 0]) == -1);
r2 = r2+cn; r2(r2>length(a1)) = length(a1);
for z3 = 1:length(r2)
   a1(r1(z3):r2(z3)) = 1;
end
sns = length(find(a1==1 & AA==3))/MM1;
spc = length(find(a1==0 & AA==0))/MM2;
sn1 = pchip([0 0.5 1], [1 sns 0], xx);
sp1 = pchip([0 0.5 1], [0 spc 1], xx);
auc_cc = polyarea([0 sp1 1 0], [1 sn1 0 0]);  
%%%%%%%%%%%%%%%%%%%%%%%
% Bootstrapping
%%%%%%%%%%%%%%%%%%%%%%%
rng(1);
ref =  round(rand(1000,MM(1))*(MM(1)-1))+1;
BB = size(ref); auc1 = zeros(1,1000); 
for ii = 1:BB(1)    
    % Concatenate annotations and decision values
    cc = []; aa=[];
    r1 = ref(ii,:);
    for jj = 1:length(r1)
        dum = dec{r1(jj)};
        drf =find(isnan(dum)==0);
        aa = [aa  sum(annotat{r1(jj)}(:,drf))];
        cc = [cc dum(drf)];
    end  
    A = aa;
    C = cc;   
    M1 = length(find(A==3));
    M2 = length(find(A==0));
    a1 = C;
    %a1 = check_s_len(a1, 10);
    r1 = find(diff([0 a1 0]) == 1);
    r2 = find(diff([0 a1 0]) == -1);
    r2 = r2+cn; r2(r2>length(a1)) = length(a1);
    for z3 = 1:length(r2)
       a1(r1(z3):r2(z3)) = 1;
    end
    sns = length(find(a1==1 & A==3))/M1;
    spc = length(find(a1==0 & A==0))/M2;
    sn1 = pchip([0 0.5 1], [1 sns 0], xx);
    sp1 = pchip([0 0.5 1], [0 spc 1], xx);
    auc1(ii) = polyarea([0 sp1 1 0], [1 sn1 0 0]);      
end

%%%%%%%%%%%%%%%%%%%%%%%%
%
% IOA stuff
%
%%%%%%%%%%%%%%%%%%%%%%%%
kf2 = zeros(1,num); kf3 = kf2; kf4 = kf2;
for z1 = 1:num
    A = []; B = [];
for ii = 1:MM(1)
    d1 = dec{ii}; d2 = d1;
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
    kf1 = estimate_kf(A');
    A2 = [A(1,:)' A(2,:)' B'];
    kf2(z1) = estimate_kf(A2);
    A3 = [A(1,:)' A(3,:)' B'];
    kf3(z1) = estimate_kf(A3);
    A4 = [A(3,:)' A(2,:)' B'];
    kf4(z1) = estimate_kf(A4);

end
% Kappa for the three human experts
kap(1) = kf1;
% optimal threshold is chosen (for highest agreement)
ref1 = find(kf2+kf3+kf4==max(kf2+kf3+kf4) , 1, 'last');
ddkap(1) = kf1-kf2(ref1);
ddkap(2) = kf1-kf3(ref1);
ddkap(3) = kf1-kf4(ref1);
d3 = cell(1, MM(1));
for ii = 1:MM(1)
    d1 = dec{ii}; d2 = d1;
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

rng(1);
ref =  round(rand(1000,MM(1))*(MM(1)-1))+1;
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
% 95% confidence interval for human experts
kap(2:3) = quantile(kf1, [0.025 0.975]);
dkap(1,:) = quantile(df1, [0.025 0.975]); % substitute E 
dkap(2,:) = quantile(df2, [0.025 0.975]); % substitute E 
dkap(3,:) = quantile(df3, [0.025 0.975]); % substitute E 
dkap_dist(1,:)=df1;
dkap_dist(2,:)=df2;
dkap_dist(3,:)=df3;
