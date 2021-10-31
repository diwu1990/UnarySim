function sig_ker_m = sig_ker_corr(x,m)
% A correlation function to aid in the creation
% of the signal kernel
%
% Nathan Stevenson
% SPRC June 2004

N = length(x);
z_nmm = [zeros(1,N+m) x zeros(1,N-m)];
z_npm = [zeros(1,N-m) x zeros(1,N+m)];
ker_nm = z_npm.*conj(z_nmm);
sig_ker_m = ker_nm(N+1:2*N);
