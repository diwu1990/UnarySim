 function d2 = check_s_len(d1, lim);
 
 r1 = find(diff([0 d1 0])==1);
 r2 = find(diff([0 d1 0])==-1);
 d2 = zeros(1,length(d1));
 q1 = find(r2-r1<lim);
 r1(q1)=0; r2(q1)=0;
 r1 = r1(r1~=0); r2 = r2(r2~=0); 
 for ii = 1:length(r1)
     d2(r1(ii):r2(ii)-1) = 1;
 end