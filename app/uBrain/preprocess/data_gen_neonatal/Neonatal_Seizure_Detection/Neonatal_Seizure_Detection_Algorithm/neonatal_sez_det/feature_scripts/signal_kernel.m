function analytic_sig_ker = signal_kernel(x);
% Time-varying auto-correlation function
%
%               analytic_sig_ker = signal_kernel(x);
% 
% This function estimates the signal kernel or time-varying
% auto-correlation function of the analytic associate of input signal x
%
% INPUTS: x - the real signal under analysis
%
% OUTPUTS: analytic_sig_ker - the signal kernel
%
% Notes, uses the sig_ker_corr.m function.
%
% Nathan Stevenson
% SPRC 2004 

N = length(x);
% Estimate the analytic associate of the signal
if mod(length(x),2) == 0
    true_X = fft(x);
    analytic_X = [true_X(1) 2.*true_X(2:N/2) true_X(N/2+1) zeros(1,N/2-1)];
    analytic_x = ifft(analytic_X);    
else
    true_X = fft(x);
    analytic_X = [true_X(1) 2.*true_X(2:ceil(N/2)) zeros(1,floor(N/2))];    
    analytic_x = ifft(analytic_X);    
end

% Estimate the signal kernel
analytic_sig_ker = zeros(N,N);
for m = -round(N/2-1):1:round(N/2-1);
    analytic_sig_ker(m+round(N/2)+1,:) = sig_ker_corr(analytic_x,m); 
end

