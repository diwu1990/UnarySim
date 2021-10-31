%Function: zero_crossing
%Description: Find the number of zero crossings in data
%Author: Stephen Faul
%Date: 10/6/2010

function zc = zero_crossing(data)
zc = sum( (data(1:end-1).*data(2:end)) < 0); %multiply adjacent samples. If result is -ve, it's a zero crossing