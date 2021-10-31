function feats=compute_features(n,format,detector,filename,hpf,Num,Den,varargin)
% This function computes features for selected algorithm.

if isequal(detector,'SDA_T')
    fs=32;len = 8;olap=4;
else
    fs=64;len = 32;olap=28;
end

if format==1

    [sig, sc,fs_orig] = read_data_montage(filename);
    disp('EDF-file red in')
    
    feats= features_per_ch(n,sig,sc,len,fs_orig(1),fs,hpf,Num,Den,olap,detector);  

elseif format==2
    
    [sig, ~, ~, fs_orig, scle]  = read_edf(filename);
    sc = scle(1);
    disp('EDF-file red in')
    
    feats= features_per_ch(n,sig,sc,len,fs_orig(1),fs,hpf,Num,Den,olap,detector);  
    
elseif format==3
    
    s=load(filename);
    x=fieldnames(s);
    eeg_data=getfield(s, x{1});
    disp('Mat-file red in')
    
    if iscell(eeg_data)==0
        rows=size(eeg_data,1);
        sig=mat2cell(eeg_data,ones(1,rows),length(eeg_data));
        sc=1;
        feats= features_per_ch(n,sig,sc,len,varargin{1},fs,hpf,Num,Den,olap,detector); 
    else
        sig=eeg_data;
        sc=1;
        feats= features_per_ch(n,sig,sc,len,varargin{1},fs,hpf,Num,Den,olap,detector); 
    end

end