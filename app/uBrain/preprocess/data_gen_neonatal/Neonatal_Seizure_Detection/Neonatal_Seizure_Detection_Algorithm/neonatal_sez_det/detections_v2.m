function [td, tdl, ref_td, tdp, fd, fdur] = detections_v2(reh, rer, ah, ar)



Q = size(rer); 
fd = 0; c2 = 1; fdur=[];
for ii = 1:Q(2)
    dum_r = ones(1,rer(2,ii)-rer(1,ii)+1);
    dum_h = ah(rer(1,ii):rer(2,ii));
    if sum(dum_r.*dum_h)==0
         fd=fd+1;
         fdur(c2) = length(dum_r);
         c2 = c2+1;
    end    
    clear dum_r dum_h
end

clear Q

Q = size(reh); 
td = 0; c1 = 1;  ref_td = []; tdl = []; tdp=[];
for ii = 1:Q(2)
    dum_h = ones(1,reh(2,ii)-reh(1,ii)+1);
    dum_r = ar(reh(1,ii):reh(2,ii));
    if sum(dum_r.*dum_h)>0
        td = td+1;
        tdl(c1) = reh(2,ii)-reh(1,ii)+1;
        tdp(c1) = sum(dum_r.*dum_h);
        ref_td(c1)=ii;
        c1=c1+1;
    end    
    clear dum_r dum_h
end


