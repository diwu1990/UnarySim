function [auc1,ref] = auc_ci_concat(dec_raw, annotat, cn,num)

% CHANGE THE SVM OUTPUT INTO A BINARY DECISION
M = size(dec_raw); MM = size(dec_raw{1}');
d4 = cell(1,M(1)); 
for ii = 1:M(1)
    a = annotat{ii};
    a = sum(a);
    dum = zeros(MM(2), length(dec_raw{ii}{1}));
    for jj = 1:MM(2)
        dd = conv(dec_raw{ii}{jj}, ones(1,3))/3;
        dum(jj, :) = dd(2:end-1);
    end

    d1 = max(dum);                  
    d1 = medfilt1(d1,3);
    d2 = [];
    for kk = 1:length(d1)
        d2 = [d2 d1(kk).*ones(1,4)];
    end
    d5=d2;
    d5(a==2)=NaN;d5(a==1)=NaN;
    d4{ii} = d5;
    val(ii,:) = [min(d2) max(d2)];
end

th = linspace(min(val(:,1)), max(val(:,2)),num);

%%%%%%
% Bootstrapping
%%%%%%
rng(1);
ref =  round(rand(1000,M(1))*(M(1)-1))+1;
BB = size(ref); auc1 = zeros(1,1000); 
for ii = 1:BB(1)
    % Concatenate annotations and decision values
    cc = []; aa=[];
    r1 = ref(ii,:);
    for jj = 1:length(r1)
        dum = d4{r1(jj)};
        drf =find(isnan(dum)==0);
        aa = [aa  sum(annotat{r1(jj)}(:,drf))];
        cc = [cc dum(drf)];
    end
    
    A = aa;
    C = cc;
    
    M1 = length(find(A==3));
    M2 = length(find(A==0));
    sens = zeros(1,num); spec = sens;
    for jj = 1:length(th)
        a1 = zeros(1,length(C));
        a1(C>th(jj)) = 1;
           r1 = find(diff([0 a1 0]) == 1);
           r2 = find(diff([0 a1 0]) == -1);
           r2 = r2+cn; r2(r2>length(a1)) = length(a1);
           for z3 = 1:length(r2)
               a1(r1(z3):r2(z3)) = 1;
           end
        sens(jj) = length(find(a1==1 & A==3))/M1;
        spec(jj) = length(find(a1==0 & A==0))/M2;
    end
    auc1(ii) = polyarea([0 spec 1 0], [1 sens 0 0]);
   
end

