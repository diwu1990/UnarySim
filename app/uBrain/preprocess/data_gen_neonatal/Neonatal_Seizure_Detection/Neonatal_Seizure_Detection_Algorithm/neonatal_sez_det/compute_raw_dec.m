function dec_raw=compute_raw_dec(norm_val,feat,SVM,n)

ss=length(feat{1}{1});
mu=norm_val(1:ss);
sig=norm_val((ss)+1:end);
dec_raw=cell(length(feat),1);
ff_dec=cell(length(feat),1);
for i=1:length(feat) 
    ff=[];
    for ii=1:length(feat{1})
        ff=[ff;feat{i}{ii}];
    end
    ff_dec{i}=ff;
end
if n~=0
    parfor i=1:length(feat)   
        % normalize features
        val=bsxfun(@rdivide,bsxfun(@minus,ff_dec{i},mu),sig);
        [ ~,dec_raw{i}] = predict(SVM,val);
        dec_raw{i}=dec_raw{i}(:,2);
    end
else
    for i=1:length(feat)   
        % normalize features
        val=bsxfun(@rdivide,bsxfun(@minus,ff_dec{i},mu),sig);
        [ ~,dec_raw{i}] = predict(SVM,val);
        dec_raw{i}=dec_raw{i}(:,2);
    end
    
end
