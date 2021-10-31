function feats=feats_sda(epoch,fs,hpf)


%%%%%%%%%%%%%%%%%
% SNLEO FEATURES
%%%%%%%%%%%%%%%%%
% 1) skewness snleo
% 2) regularity
% 3) number of spikes
% 4) width of spikes
% 5) spike gap
% 6) mean spike correlation
% 7) std spike correlation
% 8) mean snleo
% 9) std snleo
% 10) spikiness

% High-pass filtering
fc=1;
Num = [1 -1]; Den =  [1 -(1-fc*2*pi/fs)]; 
epoch1=filter(Num, Den, epoch);

f1 = zeros(1,9);
snleo = nlin_energy(epoch1,fs);
win=4*fs;
olap=fs;
f1(1) = skewness(snleo);
[~, f1(2)] = regularity(snleo,win,olap);
[snum, swidth, sgap, rho1, rho2,~,ss] = spike_analysis(snleo, epoch1, fs);
f1(3) = snum;
f1(4) = swidth;
f1(5) = sgap;
f1(6) = rho1;
f1(7) = rho2;
f1(8) = mean(snleo);
f1(9)=std(snleo);
f1(10) = ss;
feats=f1;
clear f1;

%%%%%%%%%%%%%%%%%%
% PWELCH/STFT features
%%%%%%%%%%%%%%%%%%
% 1) THD1
% 2) THD2
% 3) max THD
% 4) Relative delta power
% 5) Relative theta power
% 6) Relative alpha power
% 7) Relative beta power
% 8) max THD

% High-pass filtering
fc=0.5;
Num = [1 -1]; Den =  [1 -(1-fc*2*pi/fs)]; 
epoch2=filter(Num, Den, epoch);

[x_diff, ~] = diff_fbm(epoch2);
fref=1;    
w=1*fs; 
w2=4*fs;
s=0.5*fs;

stft = estimate_stft(x_diff,w2,w2-s);

nspec=pwelch(x_diff,w,w-s,(length(epoch2)/(fs*2))*w);
ff = estimate_thd_v3(nspec, fs, 1, fref);

f1(1)=ff(1);
f1(2)=ff(2);
f1(3)=ff(3);
NFFT =length(nspec);
thd1 = find(nspec==max(nspec));
val = nspec(thd1);
hp = find(nspec(thd1:NFFT)<=val/2,1);
lp = thd1-find(nspec(1:thd1)<=val/2,1,'last');
if isempty(hp)==1; hp=NFFT-thd1; end
if isempty(lp)==1; lp = thd1; end

N = length(nspec);
f = linspace(0,fs/2, N);

ref1 = find(f>=0.5 & f<4);
ref2 = find(f>=4 & f<8);
ref3 = find(f>=8 & f<12);
ref4 = find(f>=12 & f<30);
sp_ref = sum(nspec([ref1 ref2 ref3 ref4]).^2)/NFFT; % Spectral Power
f1(4) = sum(nspec(ref1).^2)/NFFT/sp_ref;   % Relative delta power
f1(5) = sum(nspec(ref2).^2)/NFFT/sp_ref;   % Relative theta power
f1(6) = sum(nspec(ref3).^2)/NFFT/sp_ref;   % Relative alpha power
f1(7) = sum(nspec(ref4).^2)/NFFT/sp_ref;   % Relative beta power

THD=zeros(size(stft,1),1); 
for iv=1:size(stft,1)
    ts=stft(iv,:);
    THD(iv)=max(ts);
end
f1(8)=sum(THD)/sum(sum(stft));

feats=[feats,f1];
clear f1

%%%%%%%%%%%%%%%%%%
% Median TFC
%%%%%%%%%%%%%%%%%%

% High-pass filtering
epoch3=filter(hpf(1:5),hpf(6:10),epoch);

% Smoothing windows in time and frequency domain
win_t=151;win_f=101;
M=3;
x_new=resample(epoch3,32,64);
[x_diff, ~] = diff_fbm(x_new);
[~, tf_diff] = wvd(x_diff);
ff=256;
tf2 = conv2(tf_diff, hamming(win_t)*hamming(win_f)','same');
tf2=tf2';

% Further smoothing along frequency axis
tf3=zeros(length(epoch)/2,ff);
n=0;
for iy=1:4:size(tf2,2)
    n=n+1;
    tf3(:,n)=sum(tf2(:,iy:iy+3),2);
end

% Further smoothing along time axis
tf4=zeros(length(epoch)/fs, ff);
n=0;
for ix=1:32:size(tf3,1)
    n=n+1;
    tf4(n,:)=sum(tf3(ix:ix+31,:));
end

tf=tf4;tf(tf4<0)=0;
f1 = freq_corr(tf, M);

feats=[feats f1(2)];
clear f1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Log spectrum and amplitude envelope
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1) mean spectral power (log)
% 2) amplitude envelope

% High-pass filtering
fc=1.0;
Num = [1 -1]; Den =  [1 -(1-fc*2*pi/fs)]; 
epoch4=filter(Num, Den, epoch);

dat=log(epoch4+1);
% spectral power mean
w=1*fs; 
s=0.5*fs;
nspec=pwelch(dat,w,w-s,16*w);
N = length(nspec);
f = linspace(0,fs/2, N);
ref1 = find(f>=0.5 & f<4);
ref2 = find(f>=4 & f<8);
ref3 = find(f>=8 & f<12);
ref4 = find(f>=12 & f<30);
f1(1) = sum(nspec([ref1 ref2 ref3 ref4]).^2)/N; 
% envelope amplitude
f1(2)=mean(abs(hilbert(epoch4)));
feats=[feats f1];

end
