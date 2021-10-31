function feats=features_per_ch(n,sig,sc,len, fs_orig,fs,hpf,Num,Den,olap,detector)


if n>1 && isequal(detector,'SDA')==1
    feats=feats_parfor_SDA(n,sig,sc,len,fs_orig,fs,Num,Den,olap,hpf);
    
elseif n>1 && isequal(detector,'SDA_T')==1 
    feats=feats_parfor_SDA_T(n,sig,sc,len,fs_orig,fs,Num,Den,olap);
    
elseif n>1 && isequal(detector,'SDA_DB_mod')==1 
    feats=feats_parfor_SDA_DB_mod(n,sig,sc,len,fs_orig,fs,Num,Den,olap);
    
else
    disp('Parallelization not used')
    if isequal(detector,'SDA')==1
        feats=cell(length(sig),1);
        h = waitbar(0,'Computing features');
        for i=1:length(sig)
            feats{i}=ch_sda(sig,sc,len, fs_orig,fs,Num,Den,olap,i,hpf);
            waitbar(i/length(sig),h,['Computed features on channels ',num2str(i),'/',num2str(length(sig))]);            
        end
        close(h)
    elseif isequal(detector,'SDA_T')==1 
        feats=cell(length(sig),1);
        h = waitbar(0,'Computing features');
        for i=1:length(sig)
            feats{i}=ch_temko(sig,sc,len, fs_orig,fs,Num,Den,olap,i);
            waitbar(i/length(sig),h,['Computed features on channels ',num2str(i),'/',num2str(length(sig))]);            
        end
        close(h)
    elseif isequal(detector,'SDA_DB_mod')==1 
        feats=cell(length(sig),1);
        h = waitbar(0,'Computing features');
        for i=1:length(sig)
            disp(i)
            feats{i}=ch_db(sig,sc,len, fs_orig,fs,Num,Den,olap,i);
            waitbar(i/length(sig),h,['Computed features on channels ',num2str(i),'/',num2str(length(sig))]);            
        end
        close(h)
    end
end

end
