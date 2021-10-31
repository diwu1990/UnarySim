function feats=feats_db_mod(epoch, fs)
% 1) number of spikes
% 2) width of spikes
% 3) mean spike correlation
% 4) spikiness
% 5) skewness of autocorrelation
% 6) Wavelet 1-2 Hz
% 7) Wavelet 2-4 Hz
% 8) Wavelet 4-8 Hz
% 9) Zero-crossings of autocorrelation function

% High-pass filtering
fc=1;
Num = [1 -1]; Den =  [1 -(1-fc*2*pi/fs)]; 
epoch=filter(Num, Den, epoch);

snleo = nlin_energy(epoch,fs);
[snum, swidth, ~, rho1, ~,~,ss] = spike_analysis(snleo, epoch, fs);
f1(1) = snum;
f1(2) = swidth;
f1(3) = rho1;
f1(4) = ss;

p=autocorr(epoch, length(epoch)/2);
f1(5)=skewness(p);
f1(6:8)=wavelet_db(epoch);
f1(9)= autocorr_crossings(p);

feats=f1;

end