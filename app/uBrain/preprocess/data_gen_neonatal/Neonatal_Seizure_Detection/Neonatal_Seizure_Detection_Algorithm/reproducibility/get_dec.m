function dec=get_dec(dec_raw, annotat, cn)
%%%%%%%%%%%%%%%%%%%%%%%%%
% IOA 
%
% this part selects the optimal threshold for the SDA by maximising the
% agreement with the human expert.
%%%%%%%%%%%%%%%%%%%%%%%%

% CHANGE THE SVM OUTPUT INTO A BINARY DECISION
M = size(dec_raw); MM = size(dec_raw{1});
d4 = cell(1,M(1)); 
val = zeros(M(1),2); 
dec=cell(M(1),1);
for ii = 1:M(1)
    a = annotat{ii};
    a = sum(a);
    dum = zeros(MM(2), length(dec_raw{ii}{1})); 
    for jj = 1:MM(2)
        dd = conv(dec_raw{ii}{jj}, ones(1,3))/3;
        dum(jj, :) = dd(2:end-1);
    end
    d1 = max(dum, [], 1);                  % This is my postprocessing stage here
    d1 = medfilt1(d1,3);
    d2 = [];
    for kk = 1:length(d1)
        d2 = [d2 d1(kk).*ones(1,4)];
    end
    d4{ii} = d2; 
    val(ii,:) = [min(d2) max(d2)] ;
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
    dec{ii}=a1;
end
