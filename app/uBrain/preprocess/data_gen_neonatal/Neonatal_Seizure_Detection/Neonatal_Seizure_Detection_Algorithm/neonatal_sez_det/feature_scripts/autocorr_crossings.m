function f = autocorr_crossings(p)

r1 = find([p 0]<=0 & [0 p]>0);
r2 = find([p 0] >= 0 & [0 p]<0);
if length(r1)>=4 && length(r2)>=3
    ints1 = [(r2(1)-r1(1))  (r2(2)-r1(2))  (r2(3)-r1(3))];
    ints2 = [(r1(2)-r2(1)) (r1(3)-r2(2)) (r1(4)-r2(3))];
    dum1 = [abs(ints1(2)-ints1(1))/mean(ints1(1:2)) abs(ints1(3)-ints1(2))/mean(ints1(2:3)) abs(ints1(3)-ints1(1))/mean(ints1([1 3]))];
    dum2 = [abs(ints2(2)-ints2(1))/mean(ints2(1:2)) abs(ints2(3)-ints2(2))/mean(ints2(2:3)) abs(ints2(3)-ints2(1))/mean(ints2([1 3]))];
    f = 100*mean([dum1 dum2]);
else
    if length(r1)>=3 && length(r2)>=2
    ints1 = [(r2(1)-r1(1)) (r2(2)-r1(2))];
    ints2 = [(r1(2)-r2(1)) (r1(3)-r2(2))];
    dum1 = abs(ints1(2)-ints1(1))/mean(ints1);
    dum2 = abs(ints2(2)-ints2(1))/mean(ints2);
    f = 100*mean([dum1 dum2]);
    else
    f = 100;
    end
end


end