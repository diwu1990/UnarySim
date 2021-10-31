function num = num_minmax(epoch)

num_max = length(findpeaks(epoch));
num_min = length(findpeaks(epoch.*(-1)));

num = num_max + num_min;