%
%Stephen Faul
%21st July 2004
%
%EXTRACT_SUBBANDS: returns each individual subband from the
%                  result of a wavelet transform
%
%                  input: coeffs -- the coefficient vector produced
%                                   from wt.m
%                            pos -- the positions of each of the subbands
%                                   (use sub_pos.m to produce this vector)
%                  
%                  output: bands -- A structure containing each subband. A
%                                   structure was used instead of a matrix
%                                   as the subbands are not equal lengths

function bands = extract_subbands(coeffs,pos)
pos=round(pos);
for i=1:length(pos)
    if i>1
        bands{i} = coeffs(pos(i):pos(i-1)-1);
    else
        bands{1} = coeffs(pos(1):end);
    end
end