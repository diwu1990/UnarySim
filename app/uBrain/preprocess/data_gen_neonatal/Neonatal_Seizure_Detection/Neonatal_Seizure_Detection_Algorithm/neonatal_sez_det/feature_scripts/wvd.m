function [amb, tfrep] = wvd(x);
% Wigner-Ville Distribution
%
%                  [amb, tfrep] = wvd(x);
%
% This function estimates the Wigner-Ville Distribution on signal x
%
% INPUTS: x - signal under analysis
% 
% OUTPUTS: amb - the ambiguty domain representation of x
%                      tfrep - the time-frequency representation of x (the WVD)
%
% Notes: uses the signal_kernel.m function to estimate the time-varying
%              correlation function
%
% Nathan Stevenson
% SPRC 2004

N = length(x);
analytic_sig_ker = signal_kernel(x);
tfrep = real(1./N.*fft(ifftshift(analytic_sig_ker,1), N, 1));
amb = fftshift(1./N.*fft(analytic_sig_ker, N, 2),2);