function anew = oscillatory_type(dat, fs);
% This function is a version of the oscillatory type detector in the
% algorithms of Deburchgraeve et al. 2008.
%
% Inputs: dat - a single channel of EEG
%         fs - sampling frequency
%
% Output: anew - a preliminary binary detection based on the presence of
%                oscillatory activity
%
% subfunctions - ac_analysis
%
% Nathan Stevenson
% University of Helsinki
% 4 May 2018


% Preliminary wavelet based segmenetation
% Do wavelet transform
N = wmaxlev(length(dat),'bior3.5');
[lo_d, hi_d, lo_r, hi_r] = wfilters('bior3.5');
M = length(lo_d);
n = length(dat);
% Decompose with BiOrthoganl Filter
dum1 = dat;
for ii = 1:N
        dum2 = conv(hi_d, dum1);            % Detail
        dum1 = conv(lo_d, dum1);            % Approximation
        cfs{ii} = dum2(2:2:end); 
        clear dum2
        dum1 = dum1(2:2:end);
        len(ii) = length(dum1);
end
cfs{ii+1} = dum1;
len(ii+1) = length(dum1);

C1 = [];
for ii = N:-1:1
    C1 = [C1 cfs{ii}];
end

% Reconstruction wavelet components to all have original signal length
wvr = zeros(N+1,length(dat));
lev = [1:N N];
for ii = N+1:-1:1
    if ii == N+1 || ii == N
        tlen = [n len(1:end-2) len(end-2)];
    else
        tlen = [n len(1:end-2) len(end-2)];
        tlen= [zeros(1,N-ii) tlen(1:ii)];
    end
     dum = cfs{ii};   
     for jj = 1:lev(ii);
         if jj > 1 || ii==N+1
             flt = lo_r;
         else
             flt = hi_r;
         end        
         dum1 = zeros(1,2*length(dum));
         dum1(1:2:end) = dum;
         dum = conv(flt, dum1);
         dum = dum(length(lo_r)-1:end-length(lo_r)+1-mod(tlen(N+1-jj),2));
         clear dum1
     end
 wvr(ii,:) = dum;
 clear dum   
end

% Segment signal into 3s epochs and highlight periods with high relative
% waveelt energy in certain bands
epl = 3*fs; tl = 30*fs;
block_no = floor((length(dat)-1)/epl);
annotat = zeros(1, length(dat)/fs);
for ii = 11:block_no;
    r1 = (ii-1)*epl+1; r2 = r1+epl-1; 
    r3 = (ii-1)*epl-tl+1; r4 = r1-1;
    r5 = (ii-1)*epl/fs+1; r6 = r5+epl/fs-1;
    val1 = diff(wvr(5:7, r1:r2)').^2; % 1-8Hz
    val2 = diff(wvr(5:7, r3:r4)').^2; % 1-8Hz
    v1 = quantile(val1, 0.75);
    v2 = quantile(val2, 0.75);
    vv = max(v1./v2);
    if vv>2.5
       annotat(r5:r6)=1;
    end
end

% Perform analysis of the auto-correltation function in sections
% highlighted by the wavelet segmentation
r1 = find(diff([0 annotat 0])==1);
r2 = find(diff([0 annotat 0])==-1);
ref1 = find(r2-r1>=5);
r1 = r1(ref1); r2 = r2(ref1);
anew = zeros(1,length(dat)/fs); 
for ii = 1:length(r1);
    block_no = (r2(ii)-r1(ii))-4; 
    for jj = 1:block_no;
        r3 = r1(ii)+jj-1; r4 = r3+4; % 5s epochs
        [anew(r3:r4), ~, ~] = ac_analysis(dat, fs, r3, r4); % ACF analysis
    end
end


end

