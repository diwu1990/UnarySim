function [num, pp]=find_num_benchmark(descend)
dd=zeros(3,3);
for i=1:3
    for ii=1:3
    f=find(diff(flipud(descend{i}(:,ii)<0))==-1);
    if isempty(f)
        dd(i,ii)=0;
    else
        dd(i,ii)=f(1);
    end
    end    
end

num=max(max(dd));
pp=round(num/length(descend{1}) *100);