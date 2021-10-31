function feats=get_features(group,path,hp,Num,Den,fs_orig,n)
% This function computes features for selected algorithm.
[n_patients,exam_names]=getNroOfPatients(path,'.edf');

if isequal(group,'SDA_T')
    fs=32;len = 8;olap=4;
elseif isequal(group, 'SDA_DB')
    disp('No features needed for SDA_DB')
else
    fs=64;len = 32;olap=28;
end
if isequal(group,'SDA_DB')~=1
    feats=cell(1);
    disp('Features computed for patients:')
    for p=1:n_patients
        try             
        feats{p}=read_eeg(exam_names{p},path,hp,group,len,olap,fs,Num,Den,fs_orig,n);
        catch ME
            disp(['Error feats ',num2str(p)]) 
            disp(ME)    
        end 
        disp(p)
    end  
end