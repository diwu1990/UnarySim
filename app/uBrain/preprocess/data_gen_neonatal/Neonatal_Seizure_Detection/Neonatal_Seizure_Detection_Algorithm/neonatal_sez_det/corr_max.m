function p = corr_max(cs1, cs2);
% This function determines the maximum correlation between two signal
% elements in cs1 and cs2
%

if length(cs1)==length(cs2);
     p = corr(cs1', cs2');
else
if length(cs1)>length(cs2); a = cs2; b = cs1; end
if length(cs1)<length(cs2); a = cs1; b = cs2; end
stp = length(b)-length(a);
p1 = zeros(1, stp);
for z1 = 1:stp;
   p1(z1) = corr(a',b(z1:length(a)+z1-1)');
end
p = max(p1);
 end



end

