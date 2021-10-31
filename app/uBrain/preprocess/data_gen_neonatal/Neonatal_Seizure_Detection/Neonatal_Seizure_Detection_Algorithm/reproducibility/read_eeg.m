function feat_mat=read_eeg(p,path,hpf,group,len,olap,fs,Num,Den,fs_orig,n)

% load patient data and convert to bipolar
[sig, sc,~] = read_data_montage([path,p]);

try
    feat_mat= features_per_ch(n,sig,sc,len,fs_orig,fs,hpf,Num,Den,olap,group);
catch ME
    disp(ME)
    disp(['Error patient ',num2str(p)])
end