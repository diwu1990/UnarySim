function mean_coeffs=wavelet_db(epoch)
levels=5;
[C,L]=wavedec(epoch,levels,'db3');
% Get bands
ss=L(2:4);
bb=68;
bands=cell(1);
for i=1:3   
bands{i}=C(bb:bb+ss(i));
bb=bb+ss(i);
end

%MEAN OF COEFFICIENTS AND ABSOLUTE ENERGY
for j=1:length(bands)
    mean_coeffs(j)=mean(bands{j}.^2);
end
end