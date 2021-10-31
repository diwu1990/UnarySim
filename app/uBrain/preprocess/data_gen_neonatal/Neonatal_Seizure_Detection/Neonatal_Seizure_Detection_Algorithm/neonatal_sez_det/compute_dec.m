function a1=compute_dec(dec_raw, cn, thr)

% CHANGE THE SVM OUTPUT INTO A BINARY DECISION

dum = zeros(length(dec_raw), length(dec_raw{1}));
for jj = 1:length(dec_raw)
    dd = conv(dec_raw{jj}, ones(1,3))/3;
    dum(jj, :) = dd(2:end-1);
end
d1 = max(dum);                  % This is my postprocessing stage here
d1 = medfilt1(d1,3);
d2 = [];
for kk = 1:length(d1)
    d2 = [d2 d1(kk).*ones(1,4)];
end
  
d1 = d2;

a1 = zeros(1, length(d1));
a1(d1>thr) = 1;
a1 = check_s_len(a1, 10);
r1 = find(diff([0 a1 0]) == 1);
r2 = find(diff([0 a1 0]) == -1);
r2 = r2+cn; r2(r2>length(a1)) = length(a1);
for z3 = 1:length(r2)
   a1(r1(z3):r2(z3)) = 1;
end
if length(a1)>length(d1);  a1 = a1(1:length(d1)); end


