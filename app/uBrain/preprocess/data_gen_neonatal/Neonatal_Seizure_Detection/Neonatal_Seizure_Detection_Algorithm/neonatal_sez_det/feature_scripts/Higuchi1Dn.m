function [FD,kvect,Lk] = Higuchi1Dn(X,kmax)
%Higuchi1D.m
%
%   Usage: [FD] = Higuchi1D(X,[kmax])
%
%   Input:  X:  is a 1D vector containing values of a signal. 
%           kmax:  The maximum number of samples to skip when creating the
%                  new time series... Default is N/10 ~ Max is N/4
%
%   Output: FD: is the estimated fractal dimension when using the 
%               Higuchi algorithm method.
%
%   NOTES: This is a test version of the Higuchi algorithm. It is 
%          Defined in: T. Higuchi, "Approach to an irregular time series
%                      on the basis of the fractal theory," Physica D,
%                      vol. 31, pp 277-283, 1988.
%
%
% Written by: Luke Rankine ~ December 2004
%-----------------------------------------------------------------%

N = length(X);

if nargin < 2
    kmax = floor(N/10);
end

if kmax > floor(N/4)
    error('kmax must be less than 1/4 of the length of X');
end

%% Creating the value of "k" to be used for analysis which is not evenly
%% sampled
pos = 1;
val = 11;
ktemp = 0;
kvect = [];
while ktemp < kmax
    if pos <= 4
        ktemp = pos;
        kvect = [kvect,ktemp];
    else
        ktemp = floor(2^((val-1)/4));
        val = val+1;
        kvect = [kvect,ktemp];
    end
    pos = pos + 1;
end
%%----------------------------------------------------

tempL = 0;
for k = kvect
    for m = 1:k
        finish = floor((N-m)/k);
        for i = 1:finish+1
            x(i) = X(m + (i-1)*k);            
        end % DONT FORGET TO CLEAR 'x' BEEFORE THE NEXT 'm'
        for i = 1:finish
            tempL = tempL + abs(x(i+1) - x(i));
        end
        Lm(m) = (tempL*(N-1)/(finish*k))/k;
        clear x
        tempL = 0;
    end
    Lk(k) = mean(Lm);
end
l = 1;
for k = kvect
    a(l) = log2(k);   % This is the x-coord for LSQR regression
    b(l) = log2(Lk(k)); % This is the y-coord for LSQR regression
    l = l+1;
end
% figure
% plot(a,b,'*-')

% [A,c,coef] = LSQRnorm(a,b,1); % First order polynomial regression Using
                                % My own least squares
% 
% FD = -coef(2);
A = polyfit(a,b,1); % First order polynomial regression using Matlab func.

FD = -A(1);
