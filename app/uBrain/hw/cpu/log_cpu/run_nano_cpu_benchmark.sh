#!/bin/bash

power_mode=1 # 0: max power mode, 1: low power 5W mode
num_trial=100 # number of random trial per config
time_baseline=10

# All possible frequency: "1479000 1428000 1326000 1224000 1132800 921600 710400 614400 518400 403200 307200 204000 102000"
# initialize frequency test list
if [[ $power_mode == 0 ]];
then
    list_freq="1479000 1428000 1326000 1224000 1132800 921600 710400 614400 518400 403200 307200 204000 102000" 
else
    list_freq="921600 710400 614400 518400 403200 307200 204000 102000"
fi

# initialize power mode
sudo /usr/sbin/nvpmodel -m 0
echo "1479000" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
echo "1479000" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
echo "1" > /sys/devices/system/cpu/cpu1/online
echo "1" > /sys/devices/system/cpu/cpu2/online
echo "1" > /sys/devices/system/cpu/cpu3/online
echo "begin benchmark loop"

# enter test loop
for freq_curr in $list_freq; do

	# configure core frequency
	echo $freq_curr
	sudo /usr/sbin/nvpmodel -m "$power_mode"
	echo "$freq_curr" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
	echo "$freq_curr" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq

	# shut down extra cores
	echo "0" > /sys/devices/system/cpu/cpu1/online
	echo "0" > /sys/devices/system/cpu/cpu2/online
	echo "0" > /sys/devices/system/cpu/cpu3/online
	
	# print cpu status
	jetson_clocks --show > cpu_power_stat_pm_${power_mode}_freq_${freq_curr}.log
 
	# output power stat with an interval of 500ms
	tegrastats --interval 500 | ts '[ts: %.s]' >> cpu_power_stat_pm_${power_mode}_freq_${freq_curr}.log &

	# sleep for x second
	sleep $time_baseline

	# run model test
	date +%s.%N > model_log_stat_pm_${power_mode}_freq_${freq_curr}.log
	python3 model_benchmark_fp.py --task_mi --rnn_hard -rt=$num_trial >> model_log_stat_pm_${power_mode}_freq_${freq_curr}.log
	pidModel=$!

	sleep 2
	
	# kill processes
	pkill -P $$
done

# turn cpu back online in full mode
echo "task finished, back online"
sudo /usr/sbin/nvpmodel -m 0
echo "1479000" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq
echo "1479000" > /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq
echo "1" > /sys/devices/system/cpu/cpu1/online
echo "1" > /sys/devices/system/cpu/cpu2/online
echo "1" > /sys/devices/system/cpu/cpu3/online




