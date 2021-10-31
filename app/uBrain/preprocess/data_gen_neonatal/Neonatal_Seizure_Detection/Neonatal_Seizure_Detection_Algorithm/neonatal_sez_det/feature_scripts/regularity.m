function [f2, f3] = regularity(snleo, win, olap);
% Attempts to measure 'regulatity of snleo'. In this case
% we simply look at the histrogram of the SNLEO over sub-windows and
% evaluate their correlation. We also observe the variation of the skewness
% over this time period as well.
%
% Input: snleo - smoothed nonlinear energy operator 1xN
%             win - window length of segmentatation M
%             olap - overlap of segments (actually M-overlap)
% 

N = length(snleo);
block_no = floor(N./olap) - win/olap+1;
[~,x] = hist(snleo,100);
n = zeros(block_no, length(x));
for ii = 1:block_no;
    r1 = (ii-1)*olap+1; r2=r1+win-1;
    [n(ii,:),~] = hist(snleo(r1:r2), x);
    sk(ii) = skewness(snleo(r1:r2));
end
P = corr(n', 'type', 'Spearman');
f2 = mean(P(P>0 & P<1));
f3 = std(sk(isnan(sk)~=1 & isinf(sk)~=1));
