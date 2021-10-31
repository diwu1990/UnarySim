function [y, H] = diff_fbm(x);

% estimate fractal dimension and therefore Hurst exponent
H = 2-Higuchi1Dn(x);
if H<0
    y = diff(x);
    q = y(end-15:end);
    data = iddata(q', []);
    sys = ar(data, 2);
    p = forecast(sys, data, 1);
    q1 = get(p, 'outputdata');
    y = [y q1];
else
    N = length(x);
    % generate impulse response function for fractional anti-derivative
    alpha = (2*H+1);
    h = zeros(1,N);
    h(1) = 1;
    for k = 2:N
        h(k) = (alpha/2+k-1)*(h(k-1)/k);
    end
    % deconvolve impulse response from zero padded epoch
    y = deconv([x zeros(1,N-1)], h);
end