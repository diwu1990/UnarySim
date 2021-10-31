function stft=estimate_stft(x,epl,olap)

N = length(x);
block_no = round((N/olap - (epl-olap)/olap)); NFFT = 2*epl;
stft = zeros(block_no, NFFT);
for ii = 1:block_no;
    r1 = (ii-1)*olap+1; r2 = r1+epl-1;
    stft(ii,:) = abs(fft(hamming(epl)'.*(x(r1:r2)-mean(x(r1:r2))), NFFT))./epl;
end
stft = stft(:, 1:NFFT/2);