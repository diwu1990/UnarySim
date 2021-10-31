function [dec_raw,dec]=get_decision_values(group,annotat,feats,n)

% SDA
if isequal(group,'SDA')
    cn=23;
% SDA_T
elseif isequal(group,'SDA_T')
    cn=7;
% SDA_DB_mod
elseif isequal(group,'SDA_DB_mod')
    cn=28;    
end

load(['svm_',group])

dec_raw=cell(length(feats),1);
disp('Raw decision values computed for patients:')
if n>1
    pp=gcp;
    if pp.Connected~=1
        parpool(n)
    end
    parfor p=1:length(feats)
    dec_raw{p}=get_raw_dec(norm_val,feats,SVMs,p);
    disp(p)
    end
    disp('Computing binary decision...')
    dec=get_dec(dec_raw, annotat, cn);
else
    for p=1:length(feats)
        dec_raw{p}=get_raw_dec(norm_val,feats,SVMs,p);
        disp(p)
    end
    disp('Computing binary decision...')
    dec=get_dec(dec_raw, annotat, cn);
end

end
