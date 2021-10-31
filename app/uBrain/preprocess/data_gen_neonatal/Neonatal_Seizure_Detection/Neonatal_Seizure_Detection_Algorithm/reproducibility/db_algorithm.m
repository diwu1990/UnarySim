% DB algorithm
function bb=db_algorithm(path,n)

dur_min = 8; bb = cell(79,1);
% No parallelization
if n==0
    for z0 = 1:79
        bb{z0,1}=db_get_dec(path,dur_min,z0);
    end
else
    parpool(n)
    parfor z0 = 1:79
        bb{z0,1}=db_get_dec(path,dur_min,z0);
    end    
    
end


end
