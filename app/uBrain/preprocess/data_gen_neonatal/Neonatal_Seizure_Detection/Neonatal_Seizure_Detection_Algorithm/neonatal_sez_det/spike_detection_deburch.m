function new_a = spike_detection_deburch(dat, fs);
% This functio nperforms the spike detection and correlative set
% construction component of DB algorithm 
%
% Inputs: dat - a single channel of EEG
%         fs - sampling frequency
%
% Output: newa - a preliminary binary detection based on the presence of
%                spike train activity
%
% Subfunctions - corr_max
%
% Nathan Stevenson
% University of Helsinki
% 4 May 2018


% CALCULATE SMootheed NONLINEAR ENERGY OPERATOR
xm3 = [dat 0 0 0];
xm2 = [0 dat 0 0];
xm1= [0 0 dat 0];
xm0= [0 0 0 dat];
nleo = xm1.*xm2 - xm0.*xm3;
nleo = nleo(4:end);
sm = ceil(0.12*fs);  % Smooth by 120ms
dum1 = conv(abs(nleo), ones(1,sm));
snleo = dum1(floor(sm/2):end-(floor(sm/2)));

% BREAK SIGNAL INTO 5 second epochs with a 1s shift
el = 5; olap = 1;
block_no = floor(length(snleo)/(olap*fs))-4;
segs = zeros(1, length(dat));
for ii = 1:block_no;
    r1 = 1+(ii-1)*olap*fs ; r2 = r1+el*fs-1;
    rr =  r1:r2;
    dt = snleo(rr);
    th = 0.6.*(std(dt)+quantile(dt, 0.75)); % Estimate threshold
    segs(rr(dt>th))=1;    
end
 
% Assess spikes and remove if less than 60ms
r1 = find(diff([0 segs 0])==1);
r2 = find(diff([0 segs 0])==-1)-1;
sref = find(r2-r1< floor(0.06*fs));
for jj = 1:length(sref);
    segs(r1(sref(jj)):r2(sref(jj)))=0;
end

% And spikiness is less than 7 (spikiness is evaluated on the EEG not the NLEO)
r1 = find(diff([0 segs 0])==1);
r2 = find(diff([0 segs 0])==-1)-1;
sp_ness = zeros(1,length(r1));
for jj = 1:length(r1);
    rr = r1(jj):r2(jj);
    rr1 = rr(1)-(length(rr)+1):r1(jj)-1;
    rr2 = r2(jj)+1:r2(jj)+1+length(rr);
    if min(rr1)<1; rr1 = rr2; end 
    if max(rr2)>length(segs); rr2 = rr1; end
    sp_ness(jj)= max(snleo(rr))/(mean([snleo(rr1) snleo(rr2)]));
end
sref = find(sp_ness<7);
for jj = 1:length(sref);
    segs(r1(sref(jj)):r2(sref(jj)))=0;
end

% Cluster spikes according to correlations, minimum cluster size is 6
% spikes
set_no = 0; c1 = 1; cset = cell(1); cs = cset; %cx = 1;
while sum(segs)>0  
    
   
    r1 = find(diff([0 segs 0])==1);
    r2 = find(diff([0 segs 0])==-1)-1;
    
    if length(r1)>2
    % condition for validity is correlation greater than 0.8 and temporal
    % characteristics spike gap < 20s or 40* first gap
    if set_no == 0
        cs{1,1} = dat(r1(1):r2(1)); cs{1,2} = [r1(1) r2(1)];
       cs{2,1} = dat(r1(2):r2(2)); cs{2,2} = [r1(2) r2(2)];
 
        segs(r1(1):r2(1))=0; segs(r1(2):r2(2))=0;
        p = corr_max(cs{1,1}, cs{2,1});
        gpl = r1(2)-r1(1);
        if p>0.8 && gpl<20*fs
            set_no = 2;
        else
            set_no = 0;
        end       
    else
        cs{set_no+1, 1} = dat(r1(1):r2(1)); cs{set_no+1, 2} = [r1(1) r2(1)];
        segs(r1(1):r2(1))=0; 
        for z1 = 1:length(cs)-1
            p(z1) = corr_max(cs{set_no+1,1}, cs{set_no+1-z1,1});
        end

        if 40*gpl>20*fs; tlim = 2*fs; else; tlim = 40*gpl; end
        if mean(p)>0.8 &&  r1(1)-cs{end-1,2}(1)<tlim
            set_no = set_no+1;
        else
            if set_no>=6
                cset{c1}=cs;
                c1 = c1+1;
            end
            set_no = 0;
            clear cs
        end
    end
    
    else
        segs = zeros(1, length(dat));
    end
        
end


% Generate preliminary annotation based on valid spikes
annotat = zeros(size(segs));
if isempty(cset{1})==0
    for z1 = 1:length(cset);
        csd = cset{z1};
        annotat(csd{1,2}(1):csd{end,2}(2)) = 1;
    end
end

% combine annotations if separated by less than 20s
r1 = find(diff([0 annotat 0])==1);
r2 = find(diff([0 annotat 0])==-1)-1;
rr = r1(2:end)-r2(1:end-1);
rrr = find(rr<20*fs);
for ii = 1:length(rrr)
    annotat(r2(rrr(ii)):r1(rrr(ii)+1)) = 1;
end

% Sample annotation to 1s sampling frequency
block_no = length(annotat)/fs;
new_a = zeros(1,block_no);
for ii = 1:block_no;
    r1 = (ii-1)*fs+1; r2 = r1+fs-1;
    a = annotat(r1:r2);
    if sum(a)>=fs/2;
        new_a(ii) = 1;
    end
end

end
        





