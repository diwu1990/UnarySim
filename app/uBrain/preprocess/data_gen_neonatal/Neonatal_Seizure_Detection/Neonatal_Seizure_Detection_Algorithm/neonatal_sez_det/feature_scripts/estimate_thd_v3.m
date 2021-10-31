function ff = estimate_thd_v3(nspec1, fs, w, fref)

N = length(nspec1);
f = linspace(0,fs/2, N);
fr = find(f>=fref,1)-1; 
ns1 = nspec1; 
thd1 = find(ns1==max(ns1),1); 
NFFT = 2*length(nspec1);
ns1(1:fr) = 0;      % Ignore frequency content less than fref, if peak is less than 0.5 then set measures to 0
if thd1<=fr;
    ff = zeros(1,3);
else

ref = thd1;
if (thd1+1)*2<NFFT/2;
    thd2 = find(nspec1((thd1-1)*2:(thd1+1)*2)==max(nspec1((thd1-1)*2:(thd1+1)*2)), 1)+(thd1-1)*2-1;
    q1 = thd2-thd1;
    if length(find(ns1(thd1:thd2)<ns1(thd2)))>=0.5*q1-1
    ref = [ref thd2];
    end
    if (thd1+1)*3<NFFT/2;
        thd3 = find(nspec1((thd1-1)*3:(thd1+1)*3)==max(nspec1((thd1-1)*3-1:(thd1+1)*3)), 1)+(thd1-1)*3-1;
        q2 = thd3-thd2;
        if length(find(ns1(thd2:thd3)<ns1(thd3)))>=0.5*q2-1 
            ref = [ref thd3];
        end    
    end
end
nref = []; for ii = 1:length(ref); nref = [nref ref(ii)-w:ref(ii)+w]; end 
nref = nref(nref>1 & nref<length(ns1));
ff(1) = sum(ns1(nref))./sum(ns1);     
nref1 = [thd1-w:thd1+w];
nref1 = nref1(nref1>1 & nref1<length(ns1));
ff(2) = sum(ns1(nref1))./sum(ns1);
ff(3) = log10(max(ns1));
end