function [data_mont, label, sc, fs] = read_data_montage(filename)


[dat, hdr, label, fs, scle, offs]  = read_edf(filename);
sc = scle(1);
str = cell(18,2); 
str{1,1} = 'Fp2'; str{1,2} = 'F4';     % Fp2-F4
str{2,1} = 'F4'; str{2,2} = 'C4';     % F4-C4
str{3,1} = 'C4'; str{3,2} = 'P4';     % C4-P4
str{4,1} = 'P4'; str{4,2} = 'O2';     % P4-O2
str{5,1} = 'Fp1'; str{5,2} = 'F3';     % Fp1-F3
str{6,1} = 'F3'; str{6,2} = 'C3';     % F3-C3
str{7,1} = 'C3'; str{7,2} = 'P3';     % C3-P3
str{8,1} = 'P3'; str{8,2} = 'O1';     % P3-O1
str{9,1} = 'Fp2'; str{9,2} = 'F8';     % Fp2-F8
str{10,1} = 'F8'; str{10,2} = 'T4';     % F8-T4
str{11,1} = 'T4'; str{11,2} = 'T6';     % T4-T6
str{12,1} = 'T6'; str{12,2} = 'O2';     % T6-O2
str{13,1} = 'Fp1';  str{13,2} ='F7';     % Fp1-F7
str{14,1} = 'F7'; str{14,2} = 'T3';     % F7-T3
str{15,1} = 'T3'; str{15,2} = 'T5';     % T3-T5
str{16,1} = 'T5'; str{16,2} = 'O1';     % T5-O1
str{17,1} = 'Fz'; str{17,2} = 'Cz';     % Fz-Cz
str{18,1} = 'Cz';  str{18,2} ='Pz';     % Cz-Pz

for jj = 1:18
    ref1 = zeros(1,length(dat));
    ref2 = zeros(1,length(dat));
    for ii = 1:length(dat);
        ref1(ii) = length(findstr(label{ii}', str{jj,1})); 
        ref2(ii) = length(findstr(label{ii}', str{jj,2}));
    end
    
qq1 = find(ref1==1,1);
qq2 = find(ref2==1,1);
if length(dat{qq1})~=length(dat{qq2})
    data_mont{jj} = int16(zeros(1,length(dat{1})));
else
data_mont{jj} = dat{qq1}-dat{qq2}; 
end
end

