% EEG Spectral entropy per epoch
function H = spectral_entropy_g(psd,w)

flag=0;     % flag=1, for loop version. flag=0 vectorized calculation

if(flag==1)
    for i = 1:1:w
        pdf(:,i) = psd(:,i)./(sum(psd(:,i))+eps);     
    end
elseif(flag==0)
    pdf = psd./(repmat(sum(psd,1),size(psd,1),1)+eps);     
end

H = -(sum(pdf.*log2(pdf+eps))); 