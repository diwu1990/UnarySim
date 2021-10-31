% Compute temko features
function feats = feats_temko(epoch,fs,feature_list)

use_feats = 1:55;

% High-pass filtering
fc=0.5;
Num = [1 -1]; Den =  [1 -(1-fc*2*pi/fs)]; 
epoch=filter(Num, Den, epoch);

% wawelet energy
if any(use_feats == 2)
    wavelet_energy = wavelet_coeff(epoch,fs);
end
% number of minima and maxima
 if any(use_feats == 3)
     min_max = num_minmax(epoch);
 end
% rms amplitude
if any(use_feats == 4)
    RMS_amp = rms(epoch);
end
% shannon entropy
if any(use_feats == 5)
    H_shannon = entropy(epoch);
end
% Frequency features
[pxx, F] = periodogram(epoch);
win_size = length(epoch);
freq = fs*(0:win_size/2)/win_size; 
% peak frequency
if any(use_feats == 6)
    peak_freq = find(pxx==max(pxx));
end
% hjorth parameters
if any(use_feats == 7) || any(use_feats == 8) ||  any(use_feats == 9)
    [activity, mobility, complexity] = hjorth(epoch);
end
% (nlin energy)
if any(use_feats == 1) || any(use_feats == 10)
    [Nonlinear_energy,Line_length] = nonlin(epoch,fs);
end            
% spectral entropy
if any(use_feats == 11)
    H_spectral = spectral_entropy_g(pxx,(length(pxx)));
end
% zero crossings
if any(use_feats == 12)
    Zero_crossings = sum((epoch(1:end-1).*epoch(2:end)) < 0);
end
% autoregressive modelling error 
if any(use_feats == 21)
   dat1=epoch(1:floor(end/2));
   dat2=epoch(ceil(end/2)+1:end);
   fit=ar_prediction_error_newer(dat1,dat2,9);
end
if any(use_feats == 13)
    AR1 = fit(1);
end
if any(use_feats == 14)
    AR2 = fit(2);
end
if any(use_feats == 15)
    AR3 = fit(3);
end
if any(use_feats == 16)
    AR4 = fit(4);
end
if any(use_feats == 17)
    AR5 = fit(5);
end
if any(use_feats == 18)
    AR6 = fit(6);
end
if any(use_feats == 19)
    AR7 = fit(7);
end
if any(use_feats == 20)
    AR8 = fit(8);
end
if any(use_feats == 21)
    AR9 = fit(9);
end
% total power
total_power = bandpower(epoch, fs, [0 12]);
% bandpowers: 0-2, 1-3, 2-4,...,10-12 
if any(use_feats == 23)
    [band0_2] = bandpower(epoch, fs,[0 2]);
end
if any(use_feats == 24)
    [band1_3] = bandpower(epoch, fs,[1 3]);
end
if any(use_feats == 25)
    [band2_4] = bandpower(epoch, fs,[2 4]);
end
if any(use_feats == 26)
    [band3_5] = bandpower(epoch, fs,[3 5]);
end
if any(use_feats == 27)
    [band4_6] = bandpower(epoch, fs,[4 6]);
end
if any(use_feats == 28)
    [band5_7] = bandpower(epoch, fs,[5 7]);
end
if any(use_feats == 29)
    [band6_8] = bandpower(epoch, fs,[6 8]);
end
if any(use_feats == 30)
    [band7_9] = bandpower(epoch, fs,[7 9]);
end
if any(use_feats == 31)
    [band8_10] = bandpower(epoch, fs,[8 10]);
end
if any(use_feats == 32)
    [band9_11] = bandpower(epoch, fs,[9 11]);
end
if any(use_feats == 33)
    [band10_12] = bandpower(epoch, fs,[10 12]);
end
% normalized bandpowers
if any(use_feats == 34)
    [band0_2norm] = bandpower(epoch, fs,[0 2]) ./total_power;
end
if any(use_feats == 35)
    [band1_3norm] = bandpower(epoch, fs,[1 3])./total_power;
end
if any(use_feats == 36)
    [band2_4norm] = bandpower(epoch, fs,[2 4])./total_power;
end
if any(use_feats == 37)
    [band3_5norm] = bandpower(epoch, fs,[3 5])./total_power;
end
if any(use_feats == 38)
    [band4_6norm] = bandpower(epoch, fs,[4 6])./total_power;
end
if any(use_feats == 39)
    [band5_7norm] = bandpower(epoch, fs,[5 7])./total_power;
end
if any(use_feats == 40)
    [band6_8norm] = bandpower(epoch, fs,[6 8])./total_power;
end
if any(use_feats == 41)
    [band7_9norm] = bandpower(epoch, fs,[7 9])./total_power;
end
if any(use_feats == 42)
    [band8_10norm] = bandpower(epoch, fs,[8 10])./total_power;
end
if any(use_feats == 43)
    [band9_11norm] = bandpower(epoch, fs,[9 11])./total_power;
end
if any(use_feats == 44)
    [band10_12norm] = bandpower(epoch, fs,[10 12])./total_power;
end
% spectral edge frequencies: 90, 95, 80
if any(use_feats == 45)
    [SEF90, TP] = spectral_edge(pxx,freq,0.5,12,.9);
end
if any(use_feats == 46)
    [SEF95, TP] = spectral_edge(pxx,freq,0.5,12,.95);
end
if any(use_feats == 47)
    [SEF80, TP] = spectral_edge(pxx,freq,0.5,12,.8);
end
% kurtosis
if any(use_feats == 48)
    kurt = kurtosis(epoch);
end
% skewness
if any(use_feats == 49)
    skew = skewness(epoch);
end
% singular value decomposition entropy and Fisher entropy
if any(use_feats == 50) || any(use_feats == 51)
    [svd_entropy, fisher] = inf_theory(epoch);
end
% zero crossings first and second derivative and variance first and second derivative
if any(use_feats == 52) || any(use_feats == 53) ||any(use_feats == 54) ||any(use_feats == 55)
    [ZC1d, ZC2d, V1d, V2d] = raw_analysis(epoch);
end
%%
num_feats = length(use_feats);
for i = 1:num_feats
    feat_vec(i,:) = eval(feature_list{use_feats(i)});
end
feats = feat_vec';

