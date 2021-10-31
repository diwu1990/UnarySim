function p = xcorr_ns(d1, d2, swidth);

p1 = zeros(1, swidth);
for z1 = 1:swidth
    dum1 = d2(z1+1:end);
    M = min([length(dum1) length(d1)]);
    if M>5;
    dum1 = dum1(1:M);
    dum2 = d1(1:M);
    p1(z1) = corr(dum1', dum2');    
    end
end

p2 = zeros(1, swidth);
for z1 = 1:swidth
    dum1 = d1(z1+1:end);
    M = min([length(dum1) length(d2)]);
    if M>5;
    dum1 = dum1(1:M);
    dum2 = d2(1:M);
    p2(z1) = corr(dum1', dum2');    
    end
end
p = [p2(length(p2):-1:1) p1];