function [out, f1, f2] = ac_analysis(dat, fs, r3, r4);
% This function analyses the autocorrelation of a segment of EEG data
% 
% Input - dat - EEG data
%         fs - sampling frequency
%         r3 - start of 5s segement
%         r4 - end of 5s segement
%
% Ouput - out = 
%         f1 = skewness of the ACF feature
%         f2 = difference in ZCs feature
%
% Nathan Stevenson
% University of Helsinki
% 4 May 2018


epch = dat(r3*fs:r4*fs-1);
[p, l] = xcorr(epch, 'coeff');
p = p(l>=0);
f1 = skewness(p); % Skewness of the ACF

% Estimate lags between ZCs
r1 = find([p 0]<=0 & [0 p]>0); % find downwards ZC
r2 = find([p 0] >= 0 & [0 p]<0); % find upwards ZC
% calculate average differences between extrema
if length(r1)>=4 && length(r2)>=3
    ints1 = [(r2(1)-r1(1))  (r2(2)-r1(2))  (r2(3)-r1(3))];
    ints2 = [(r1(2)-r2(1)) (r1(3)-r2(2)) (r1(4)-r2(3))];
    dum1 = [abs(ints1(2)-ints1(1))/mean(ints1(1:2)) abs(ints1(3)-ints1(2))/mean(ints1(2:3)) abs(ints1(3)-ints1(1))/mean(ints1([1 3]))];
    dum2 = [abs(ints2(2)-ints2(1))/mean(ints2(1:2)) abs(ints2(3)-ints2(2))/mean(ints2(2:3)) abs(ints2(3)-ints2(1))/mean(ints2([1 3]))];
    f2 = 100*mean([dum1 dum2]);
else
    if length(r1)>=3 && length(r2)>=2
        ints1 = [(r2(1)-r1(1)) (r2(2)-r1(2))];
        ints2 = [(r1(2)-r2(1)) (r1(3)-r2(2))];
        dum1 = abs(ints1(2)-ints1(1))/mean(ints1);
        dum2 = abs(ints2(2)-ints2(1))/mean(ints2);
        f2 = 100*mean([dum1 dum2]);  
    else
        f2 = 100;
    end
end

% determine if epoch has low skew and low differences in ZC intervals
if f1<0.4 && f2 < 6
    out = 1;
else
    out = 0;
end

out = out.*ones(1,length(r3:r4));

end

