import os
import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib
import re
import pandas as pd
from numpy import mean, median

def data_parse(dir_root = "E:/ubrain_local/benchmark_parser/log_cpu/", power_mode = 1, area_fix = 3.0, win_len = 10, freq_sample = 128):
    ## Power mode: 0 - max power mode, 1 - 5W low power mode
    ## Use fix CPU area reported by
    # 1. 3mm2 -- 20nm, 1.66 mm2 -- 16nm A57 from https://en.wikichip.org/wiki/arm_holdings/microarchitectures/cortex-a57
    # 2. Tegra X1 and X1+ https://en.wikipedia.org/wiki/Tegra#Tegra_X1

    # Initialize frequency configuration list
    if power_mode == 0:
        freq_list = [1479000, 1428000, 1326000, 1224000, 1132800, 921600, 710400, 614400, 518400, 403200, 307200, 204000, 102000]
    else:
        freq_list = [921600, 710400, 614400, 518400, 403200, 307200, 204000, 102000]
    delay_window = 1.0 / freq_sample * win_len
    runtime_list = []
    power_list = []
    area_list = [area_fix] * len(freq_list)
    for idx_config, freq_curr in enumerate(freq_list):
        # get current file name
        power_file_name = "cpu_power_stat_pm_" + str(power_mode) + "_freq_" + str(freq_curr) + ".log"
        runtime_file_name = "model_log_stat_pm_" + str(power_mode) + "_freq_" + str(freq_curr) + ".log"
        # get path of the file
        path_power_file = os.path.join(dir_root, power_file_name)
        path_runtime_file = os.path.join(dir_root, runtime_file_name)
        # read runtime file
        with open(path_runtime_file) as f_runtime:
            f_runtime = f_runtime.readlines()
        ts_py_start = float(f_runtime[0]) # get the time when python start, take the power samples before it as baseline
        for line_curr in f_runtime:
            if "---Start" in line_curr:
                # get all floating number in the string
                list_model_stats = re.findall("\d+\.\d+", line_curr)
                break
        ts_model_start, ts_model_end, time_cpu_runtime = float(list_model_stats[0]), float(list_model_stats[1]), float(list_model_stats[3])
        runtime_list.append(time_cpu_runtime)
        # read power file
        df_power = pd.DataFrame(columns = ['timestamp', 'cpu_power', 'gpu_power', 'ram_power', 'total_power'])
        with open(path_power_file) as f_power:
            f_power = f_power.readlines()
        for line_curr in f_power:
            if "[ts:" in line_curr:
                ts_curr = float(re.findall("\d+\.\d+", line_curr)[0]) # get timestamp
                cpu_power_curr = float(re.search('POM_5V_CPU (\d+)', line_curr, re.IGNORECASE).group(1)) # get current cpu power in mW
                gpu_power_curr = float(re.search('POM_5V_GPU (\d+)', line_curr, re.IGNORECASE).group(1)) # get current gpu power in mW
                total_power_curr = float(re.search('POM_5V_IN (\d+)', line_curr, re.IGNORECASE).group(1)) # get current total power in mW
                ram_power_curr = total_power_curr - gpu_power_curr - cpu_power_curr # estimate current ram power
                df_power = df_power.append({'timestamp': ts_curr, 'cpu_power': cpu_power_curr, 'gpu_power': gpu_power_curr,
                                            'ram_power': ram_power_curr, 'total_power': total_power_curr   }, ignore_index=True)
        df_power_idle = df_power.loc[df_power['timestamp'] < ts_py_start]
        df_power_active = df_power.loc[(df_power['timestamp'] > ts_model_start) & (df_power['timestamp'] < ts_model_end)]
        cpu_power_idle = df_power_idle['cpu_power'].mean()
        ram_power_idle = df_power_idle['ram_power'].mean()
        cpu_power_active = df_power_active['cpu_power'].mean()
        ram_power_active = df_power_active['ram_power'].mean()
        power_list.append([cpu_power_active, ram_power_active, cpu_power_idle, ram_power_idle])

    area_list = area_list # mm^2
    runtime_list = [freq * 1000. for freq in runtime_list] # ms
    freq_list = [freq / 1000. for freq in freq_list] # MHz
    power_list = power_list # mW

    return area_list, runtime_list, freq_list, power_list


if __name__ == '__main__':
    area_list, runtime_list, freq_list, power_list = data_parse(dir_root="/home/diwu/Dropbox/project/UnaryComputing/2021 uBrain/result/mi/log_cpu/",
                power_mode=0)
    print(area_list)
    print(runtime_list)
    print(power_list)
    print(freq_list)

