function [snum, swidth, sgap, rho1, rho2,len,spikiness] = spike_analysis(snleo, dat, fs)

% Analysis output of SNLEO, generates binary annotation of SNLEO using
% adaptive threshold (based on Spearman's correlation), and then estimates
% several parameters from this annotation
%
%  Input:  snleo - smoothed nonlinear energy operator output
%               fs - sampling frequency
%
% Ouput: snum - number of candidate 'spikes'
%        swidth - spike width in ms (median and IQR)
%        sgap - inter-spike interval in ms (median and IQR)
%        rho1 - mean correlation between spikes (up to 10 spikes)
%        rho2 - STD correlation between spikes (up to 10 spikes)
%        len - spike train length
%        spikiness

%        by Nathan Stevenson

N = length(snleo);
th = linspace(quantile(snleo, 0.1), quantile(snleo, 0.9), 50);
erf1 = zeros(1, length(th)-1);
for ii = 1:length(th)-1;
    dum = zeros(1, N);
    dum(snleo>th(ii)) = 1;
    r1 = find(diff([0 dum 0]) == 1);
    erf1(ii) = length(r1);
end

q1 = find([erf1 0]<[0 erf1]);
if isempty(q1)==1
    th1 = th(50);
else
    [q2, q3]= sort(diff(q1), 'descend');
    q4 = find(erf1(q1(q3))>1, 1);
    if isempty(q4)==0; 
        th1 = th(q1(q3(q4))); 
    else
        th1 = th(50);
    end
end
dum = zeros(1, N);
dum(snleo>th1) = 1;

r1 = find(diff([0 dum 0]) == 1);
r2 = find(diff([0 dum 0]) == -1);
qq = find(r2-r1>3);
r1 = r1(qq); r2 = r2(qq);

swidth = median(r2-r1);
snum = length(r1);

if snum>1
sgap = median(r1(2:end)-r2(1:end-1));   
else
    sgap = length(dat)/2; 
end

% spike train length
if isempty(r2)==1 || isempty(r1)==1 
    len=0;
else
    len=r2(end)-r1(1);
end

if length(r1)<=3;
    rho1 = 0; rho2 = 0; spikiness=max(snleo)/mean(snleo);
else

    rx1=round(r2(1:end-1)+(r1(2:end)-r2(1:end-1))./2);
    
    % spikiness
    s=[];
    for i=1:length(rx1)-1 
        peaks=max(snleo(r1(i+1):r2(i+1)));
        bg=[snleo(rx1(i):r1(i+1)),snleo(r2(i+1):rx1(i+1))];
        s=[s,peaks./mean(bg)];
    end
    spikiness=mean(s);

    r = NaN*ones(length(rx1), length(rx1)-2);
    for z1 = 1:length(rx1)-1
        d1 = dat(rx1(z1):rx1(z1+1)); 
        if length(rx1)>z1+11
            lim1 = z1+11;
        else
            lim1 = length(rx1);
        end
        for z2 = z1+1:lim1-1;
            d2 = dat(rx1(z2):rx1(z2+1));
            p = xcorr_ns(d1, d2, floor(swidth+sgap));
            r(z1, z2-1) = max(p);
        end   
    end

    if length(rx1)>5;
        rr= [];
        for ii = 0:4
            rr = [rr ; diag(r, ii)];
        end
    else
        rr = r(isnan(r)==0);
    end
    rho1 = mean(rr);
    rho2 = std(rr);
end 


swidth = swidth/fs*1000;
sgap = sgap/fs*1000;


end

