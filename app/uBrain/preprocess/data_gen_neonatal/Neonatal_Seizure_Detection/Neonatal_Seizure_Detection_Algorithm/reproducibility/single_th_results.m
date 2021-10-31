function [sens, spec, ppv, npv,auc] = single_th_results(dec_raw, annotat, cn)

M = size(dec); MM = size(dec{1}');
% CHANGE THE SVM OUTPUT INTO A BINARY DECISION
d4 = cell(1,M(1)); acm = zeros(1,M(1)); auc = acm; %th1 = zeros(1,79);
val = [];
for ii = 1:M(1)
    a = annotat{ii};
    a = sum(a);
   dum = zeros(MM(2), length(dec_raw{ii}{1}));
   for jj = 1:MM(2)
        dd = conv(dec_raw{ii}{jj}, ones(1,3))/3;
        dum(jj, :) = dd(2:end-1);
    end
    d1 = max(dum);                  % This is my postprocessing stage here
    d1 = medfilt1(d1,3);
    d2 = [];
    for kk = 1:length(d1)
        d2 = [d2 d1(kk).*ones(1,4)];
    end
     d4{ii} = d2; 
     val = [val min(d2) max(d2)];
end

% step through a bunch of thresholds and find which one maximises the
% agreement between the algorithm and the huma raters across the entire
% annotation
th = linspace(min(min(val)), max(max(val)),1000); 
     AA = []; C = [];
for ii = 1:M(1)
    a = annotat{ii};
    a = sum(a);
    d1 = d4{ii};
    a = a(isnan(d1)==0);
    d1 = d1(isnan(d1)==0);    
   
    if length(a)>length(d1);  a = a(1:length(d1)); end
    a1 = a;   
    AA = [AA a1];
    C = [C d1];
end

kf1 = zeros(1,length(th)); 
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

    AAA = [A' B'];
    kf1(z1) = estimate_kf(AAA);

end

% pick the threshold that maximises agreement over all babies.
ref1 = find(kf1==max(kf1) , 1, 'last');
thr = th(ref1); 

% evaluate results per baby
xx = linspace(0,1,1000);
for ii = 1:M(1)
   
    a1 = annotat{ii};
    a = sum(a1);
    
    d1 = d4{ii};
    d1 = d1(isnan(d1)==0);    
    a = a(isnan(d1)==0);
   
    if length(a)>length(d1);  a = a(1:length(d1)); end
    a1 = zeros(1, length(d1));
    a1(d1>thr) = 1;
    a1 = check_s_len(a1, 10);
    r1 = find(diff([0 a1 0]) == 1);
    r2 = find(diff([0 a1 0]) == -1);
    r2 = r2+cn; r2(r2>length(a1)) = length(a1);
    for z3 = 1:length(r2)
       a1(r1(z3):r2(z3)) = 1;
    end
    if length(a1)>length(d1);  a1 = a1(1:length(d1)); end

    AA = a;
    
    M1 = length(find(AA==3));
    M2 = length(find(AA==0));

    if isempty(find(a1==1 & AA==3))
        sens(ii)=0;ppv(ii)=0;
    else
    sens(ii) = length(find(a1==1 & AA==3))/M1; % this is also recall
    ppv(ii) = length(find(a1==1 & AA==3))/length(find(a1==1)); %precision (ppv)
    end
    spec(ii) = length(find(a1==0 & AA==0))/M2; % specificity
    npv(ii) = length(find(a1==0 & AA==0))/length(find(a1==0)); %npv

    % single threshold AUC
    sn1 = pchip([0 0.5 1], [1 sens(ii) 0], xx);
    sp1 = pchip([0 0.5 1], [0 spec(ii) 1], xx);
    auc(ii) = polyarea([0 sp1 1 0], [1 sn1 0 0]); 


end



