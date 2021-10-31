function [vv1, vv2, vv3,dist] = descend_agree_DBorig(dec, annotat, cn)
M = size(dec);
dist=cell(M(1),1);
kf2 = zeros(1,1000); kf3 = kf2; kf4 = kf2;
for z1 = 1:1000
    A = []; B = [];
for ii = 1:M(1)
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
    for jj = 1:3
    a(jj,:) = check_s_len(a(jj,:), 8); 
    end
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
pa = zeros(3, M(1));

d3 = cell(1,M(1));
    for ii = 1:M(1)
          d1 = dec{ii}; d2 = d1; 
          r1 = find(diff([0 d2 0]) == 1);
          r2 = find(diff([0 d2 0]) == -1);
          r2 = r2+cn; r2(r2>length(d2)) = length(d2);
          for z3 = 1:length(r2)
              d2(r1(z3):r2(z3)) = 1;
          end
           d3{ii} = d2;
           d2 = d2(isnan(d1)==0);
          if length(d2)>length(a);  d2 = d2(1:length(a)); end
        
          for z0 = 1:3 
          a = annotat{ii}(z0,:);
          a = a(:,isnan(d1)==0);
          if length(a)>length(d2);  a = a(:,1:length(d2)); end
          a = check_s_len(a, 8); 
          pa(z0, ii) = (length(find(a==1 & d2 == 1)) + length(find(a==0 & d2 == 0))) / length(d2);
          end
       
    end


nref1 = 1:M(1); 
   
    for jj = 1:length(nref1)
        nr1 = nref1;
        nr1(jj) = 0;
        nr1 = nr1(nr1>0);
        
        C1 = []; D1 = []; E1 = []; F1 = []; 
        for kk = 1:length(nr1)
              
              dum1 = d3{nr1(kk)};
              drf =find(isnan(dum1)==0);
              C1 = [C1  annotat{nr1(kk)}(1,drf)];
              D1 = [D1 annotat{nr1(kk)}(2,drf)];
              E1 = [E1 annotat{nr1(kk)}(3,drf)];
              F1 = [F1 dum1(drf)];
              
        end   
   
    A = [C1' D1' F1'];
    kf1(jj) = estimate_kf(A);
     A = [C1' E1' F1'];
    kf2(jj) = estimate_kf(A);
     A = [D1' E1' F1'];
    kf3(jj) = estimate_kf(A);
    end

[~, idx1] = sort(kf1, 'descend');
[~, idx2] = sort(kf2, 'descend');
[~, idx3] = sort(kf3, 'descend');




qq = 1:M(1); bn = 1000;
idx = [idx1(1:M(1)) ; idx2(1:M(1)) ; idx3(1:M(1))]; vv1 = cell(1,3); vv2 = vv1; vv3 = vv1;
for z2 = 1:3

    val1 = zeros(M(1),3); val2 = val1; val3 = val1;
   
for z0 = 1:M(1)
    disp(z0)
    rng(1)
    
    nref = qq; nref(idx(z2, 1:z0))=0;
    nref = nref(nref>0); MM = length(nref);
    ref = round(rand(bn, MM)*(MM-1))+1;
    df1 = zeros(1,bn); df2 = df1; df3 = df1;
    for z1 = 1:bn
        
        C = []; D = []; E = []; F = []; 
    r1 = nref(ref(z1,:));
    for jj = 1:length(r1)
        dum = d3{r1(jj)};
        drf =find(isnan(dum)==0);
        C = [C  annotat{r1(jj)}(1,drf)];
        D = [D annotat{r1(jj)}(2,drf)];
        E = [E  annotat{r1(jj)}(3,drf)];
        F = [F dum(drf)];
    end   
   
    A = [C' D' E'];
    kf1(z1) = estimate_kf(A);
    A = [C' D' F'];
    kf2 = estimate_kf(A);
    A = [C' F' E'];
    kf3 = estimate_kf(A);
    A = [F' D' E'];
    kf4 = estimate_kf(A);

    
     df1(z1) = kf1(z1)-kf2;
     df2(z1) = kf1(z1)-kf3;
     df3(z1) = kf1(z1)-kf4;

    end
    dist{z0} = [df1;df2;df3];
    val2(z0,:) = [quantile(df1, 0.025) quantile(df2, 0.025) quantile(df3, 0.025)];
    val1(z0,:) = [mean(df1) mean(df2) mean(df3)];
    val3(z0,:) = [quantile(df1, 0.975) quantile(df2, 0.975) quantile(df3, 0.975)];
end
vv1{z2} = val1;
vv2{z2} = val2;
vv3{z2} = val3;
end


