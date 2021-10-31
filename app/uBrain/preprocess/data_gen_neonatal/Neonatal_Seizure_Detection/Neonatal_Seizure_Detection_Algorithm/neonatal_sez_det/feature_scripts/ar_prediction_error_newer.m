
function fit_error=ar_prediction_error_newer(model_data,test_data,ar_order)
fit_error = zeros(1,ar_order);
for zzz=1:ar_order
           Pxx1 = pyulear(model_data, zzz);
           if isnan(sum(Pxx1))==1;
               Pxx1 = zeros(length(Pxx1),1);
           end
           Pxx2 = pwelch(test_data);
           err = mean(abs(Pxx1-Pxx2));
           meanerr = max([mean(Pxx2) mean(Pxx1)]);
           fit_error(zzz) =  100*err/meanerr;
end
end
