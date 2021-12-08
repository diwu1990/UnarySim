from UnarySim.app.uBrain.hw.cpu.parser_benchmark import data_parse as cpu_report
from UnarySim.app.uBrain.hw.systolic_array.data_extract import data_extract as systolic_report
from UnarySim.app.uBrain.hw.data_extract.parse_xlsx import query_workbook as sc_ubrain_report
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from openpyxl import load_workbook
from brokenaxes import brokenaxes
import matplotlib.gridspec as gridspec

def bci_hw_report(cpu_dir="/home/diwu/Dropbox/project/UnaryComputing/2021 uBrain/result/mi/log_cpu/",
                    systolic_dir="/home/diwu/Project/UnarySim/app/uBrain/hw/systolic_array/result/",
                    sc_ubrain_path="/home/diwu/Project/UnarySim/app/uBrain/hw/data_extract/",
                    sc_ubrain_file="uBrain_resource.xlsx",                    
                    output_path="/home/diwu/Project/UnarySim/app/uBrain/hw/result_fig/"):

    font = {'family':'Times New Roman', 'size': 6}
    matplotlib.rc('font', **font)

    # sensor of baselines and ubrain
    area_sense_bl = 0.096
    power_sense_bl = 0.0032
    area_sense_ubr = 0.007816
    power_sense_ubr = 0.001218490376

    # extract cpu and systolic result
    # format [cpu, dram]
    area_cpu, runtime_cpu, freq_cpu, power_cpu = cpu_report(dir_root=cpu_dir, power_mode=0)
    # 4GB lpddr4 area is scaled from 25nm to 22nm, from https://ieeexplore.ieee.org/document/6901299?arnumber=6901299
    # cpu area is scaled from 20nm to 32nm, from https://en.wikichip.org/wiki/arm_holdings/microarchitectures/cortex-a57
    # format [sense, dram, cpu]
    area_cpu = [area_sense_bl, 88.1 * 4 * (22 / 25)**2, 15.85 * ((32 / 20)**2)]
    power_cpu = [[power_sense_bl, entry[1], entry[0]] for entry in power_cpu]

    # extract cpu and systolic result
    # format [dram, sram, systolic, total]
    area_sys, runtime_sys, freq_sys, power_sys = systolic_report(dir_root=systolic_dir)
    # format [sense, dram, cpu]
    area_sys = [area_sense_bl, area_sys[0], area_sys[1] + area_sys[2]]
    power_sys = [[power_sense_bl, entry[0], entry[1] + entry[2]] for entry in power_sys]

    # dram result for unary design
    area_unary_dram = 5.437757952 # mm^2, from https://github.com/diwu1990/UnarySim/blob/master/app/uBrain/hw/systolic_array/result/eyeriss_08b_bp_001c_uBrain_ddr3_w__spm/simEffOut/eyeriss_08b_bp_001c_uBrain_ddr3_w__spm_area.csv
    power_unary_dram_dynamic = 295367.4759928 / (1/128) / 10**6 # mW, energy in nJ from https://github.com/diwu1990/UnarySim/blob/master/app/uBrain/hw/systolic_array/result/eyeriss_08b_bp_001c_uBrain_ddr3_w__spm/simEffOut/eyeriss_08b_bp_001c_uBrain_ddr3_w__spm_energy.csv
    power_unary_dram_leakage = 356.3135352 # mW, from https://github.com/diwu1990/UnarySim/blob/master/app/uBrain/hw/systolic_array/result/eyeriss_08b_bp_001c_uBrain_ddr3_w__spm/simEffOut/eyeriss_08b_bp_001c_uBrain_ddr3_w__spm_power.csv
    power_unary_dram = power_unary_dram_dynamic + power_unary_dram_leakage
    
    workbook = load_workbook(filename=sc_ubrain_path+sc_ubrain_file, data_only=True)

    # extract stochastic result
    design = "sc"
    item = "TOTAL" # for TOTAL area and power
    # no halving logic
    area_sto_h0 = [area_sense_bl, area_unary_dram]
    power_sto_h0 = [power_sense_bl, power_unary_dram]
    h0_layers = ['conv1-F1', 'conv2-F1', 'fc3-F256', 'rnn4-F1-ugemm', 'fc5-F1', 'fc6-F1']
    area_sto_h0_onchip, power_sto_h0_onchip = design_area_power_extract(h0_layers, design, item, workbook)
    area_sto_h0.append(area_sto_h0_onchip)
    power_sto_h0.append(power_sto_h0_onchip)
    # print(area_sto_h0, power_sto_h0)

    # best area result
    area_sto_ba = [area_sense_bl, area_unary_dram]
    power_sto_ba = [power_sense_bl, power_unary_dram]
    ba_layers = ['conv1-F16', 'conv2-F32', 'fc3-F256', 'rnn4-F1-ugemm', 'fc5-F1', 'fc6-F1']
    area_sto_ba_onchip, power_sto_ba_onchip = design_area_power_extract(ba_layers, design, item, workbook)
    area_sto_ba.append(area_sto_ba_onchip)
    power_sto_ba.append(power_sto_ba_onchip)
    # print(area_sto_ba, power_sto_ba)

    # best power result
    area_sto_bp = [area_sense_bl, area_unary_dram]
    power_sto_bp = [power_sense_bl, power_unary_dram]
    bp_layers_sto = ['conv1-F8', 'conv2-F16', 'fc3-F256', 'rnn4-F1-ugemm', 'fc5-F1', 'fc6-F1']
    area_sto_bp_onchip, power_sto_bp_onchip = design_area_power_extract(bp_layers_sto, design, item, workbook)
    area_sto_bp.append(area_sto_bp_onchip)
    power_sto_bp.append(power_sto_bp_onchip)
    # print(area_sto_bp, power_sto_bp)

    # extract ubrain result
    design = "ubrain"
    item = "TOTAL" # for TOTAL area and power
    # no halving logic
    area_ubr_h0 = [area_sense_ubr, area_unary_dram]
    power_ubr_h0 = [power_sense_ubr, power_unary_dram]
    h0_layers = ['conv1-F1', 'conv2-F1', 'fc3-F256', 'rnn4-F1-ugemm', 'fc5-F1', 'fc6-F1']
    area_ubr_h0_onchip, power_ubr_h0_onchip = design_area_power_extract(h0_layers, design, item, workbook)
    area_ubr_h0.append(area_ubr_h0_onchip)
    power_ubr_h0.append(power_ubr_h0_onchip)
    # print(area_ubr_h0, power_ubr_h0)

    # best area result
    area_ubr_ba = [area_sense_ubr, area_unary_dram]
    power_ubr_ba = [power_sense_ubr, power_unary_dram]
    ba_layers = ['conv1-F16', 'conv2-F32', 'fc3-F256', 'rnn4-F1-ugemm', 'fc5-F1', 'fc6-F1']
    area_ubr_ba_onchip, power_ubr_ba_onchip = design_area_power_extract(ba_layers, design, item, workbook)
    area_ubr_ba.append(area_ubr_ba_onchip)
    power_ubr_ba.append(power_ubr_ba_onchip)
    # print(area_ubr_ba, power_ubr_ba)

    # best power result
    area_ubr_bp = [area_sense_ubr, area_unary_dram]
    power_ubr_bp = [power_sense_ubr, power_unary_dram]
    bp_layers_ubr = ['conv1-F4', 'conv2-F8', 'fc3-F256', 'rnn4-F1-ugemm', 'fc5-F1', 'fc6-F1']
    area_ubr_bp_onchip, power_ubr_bp_onchip = design_area_power_extract(bp_layers_ubr, design, item, workbook)
    area_ubr_bp.append(area_ubr_bp_onchip)
    power_ubr_bp.append(power_ubr_bp_onchip)
    # print(area_ubr_bp, power_ubr_bp)



    gry = "#AAAAAA"
    sal = "#FF7F7F"
    orc = "#7A81FF"
    lav = "#D783FF"
    gr1 = "#888888"
    gr2 = "#666666"

    # total area plot
    my_dpi = 300
    fig_h = 1.3
    fig_w = 3.3115
    x_axis = ["CPU", "Systolic", "SC", "SC-A", "SC-P", "uBrain", "uBrain-A", "uBrain-P"]
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    x_idx = np.arange(len(x_axis))
    width = 0.2
    area_list_2d = [area_cpu, area_sys, area_sto_h0, area_sto_ba, area_sto_bp, area_ubr_h0, area_ubr_ba, area_ubr_bp]
    area_list_2d = np.array(area_list_2d).T

    ax.bar(x_idx-width, area_list_2d[0], width, hatch = None, alpha=0.99, color=gry, label='Sense')
    ax.bar(x_axis, area_list_2d[1], width, hatch = None, alpha=0.99, color=orc, label='Store')
    ax.bar(x_idx+width, area_list_2d[2], width, hatch = None, alpha=0.99, color=sal, label='Compute')
    plt.yscale("log")

    fig.tight_layout()
    ax.set_ylabel('Area (mm$\mathregular{^2}$)')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_axis)
    # ax.set_xticklabels(x_axis, rotation=30)
    # y_idx = [0.1, 1., 10., 100.]
    # ax.set_yticks(y_idx)
    ax.legend(loc="upper right", ncol=3, frameon=True)
    plt.savefig(output_path+"/Area_total.pdf", bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)
    print("Total area fig saved!\n")


    # total power plot
    my_dpi = 300
    fig_h = 1.2
    fig_w = 3.3115
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    x_idx = np.arange(80, 170, 10)
    width = 0.5
    
    runtime_unary = 14 / 128 * 1000
    runtime_window = 10 / 128 * 1000
    line_width = 0.5
    marker_size = 1.5
    ax.plot(np.array(runtime_cpu[0:10]) + runtime_window, np.sum(power_cpu[0:10], axis=1), "-^", alpha=0.99, color=gry, label='CPU', lw=line_width, ms=marker_size)
    ax.plot(np.array(runtime_sys) + runtime_window, np.sum(power_sys, axis=1), "-P", alpha=0.99, color=lav, label='Systolic', lw=line_width, ms=marker_size)
    ax.plot(runtime_unary, np.sum(power_sto_h0), "o", alpha=0.99, color=orc, label='SC', lw=line_width, ms=marker_size)
    ax.plot(runtime_unary, np.sum(power_sto_ba), "*", alpha=0.99, color=orc, label='SC-A', lw=line_width, ms=marker_size)
    ax.plot(runtime_unary, np.sum(power_sto_bp), ">", alpha=0.99, color=orc, label='SC-P', lw=line_width, ms=marker_size)
    ax.plot(runtime_unary, np.sum(power_ubr_h0), "o", alpha=0.99, color=sal, label='uBrain', lw=line_width, ms=marker_size)
    ax.plot(runtime_unary, np.sum(power_ubr_ba), "*", alpha=0.99, color=sal, label='uBrain-A', lw=line_width, ms=marker_size)
    ax.plot(runtime_unary, np.sum(power_ubr_bp), ">", alpha=0.99, color=sal, label='uBrain-P', lw=line_width, ms=marker_size)
    plt.yscale("log")
    
    fig.tight_layout()
    ax.set_ylabel('Power (mW)')
    # ax.set_ylim((0, 4500))
    # ax.set_yticks((0, 1000, 2000, 3000))
    ax.set_xlabel('Latency ($ms$)')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_idx)
    ax.legend(loc="best", ncol=3, frameon=True)
    plt.savefig(output_path+"/Power_total.pdf", bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)
    print("Total power fig saved!\n")


    # on-chip best power plot
    my_dpi = 300
    fig_h = 1
    fig_w = 3.3115
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    x_axis = ["CPU", "Systolic", "SC", "SC-A", "SC-P", "uBrain", "uBrain-A", "uBrain-P"]
    x_idx = np.arange(len(x_axis))
    width = 0.5
    
    runtime_unary = 14 / 128 * 1000
    runtime_window = 10 / 128 * 1000
    line_width = 0.5
    marker_size = 1
    best_onchip_power_list = []
    best_onchip_power_list.append(power_cpu[9][2])
    best_onchip_power_list.append(power_sys[9][2])
    best_onchip_power_list.append(power_sto_h0[2])
    best_onchip_power_list.append(power_sto_ba[2])
    best_onchip_power_list.append(power_sto_bp[2])
    best_onchip_power_list.append(power_ubr_h0[2])
    best_onchip_power_list.append(power_ubr_ba[2])
    best_onchip_power_list.append(power_ubr_bp[2])
    ax.bar(x_idx, best_onchip_power_list, width, hatch = None, alpha=0.99, color=sal)
    
    fig.tight_layout()
    ax.set_ylabel('Power (mW)')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_axis)
    # ax.legend(loc="best", ncol=3, frameon=True)
    plt.savefig(output_path+"/Power_onchip.pdf", bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)
    print("Onchip power fig saved!\n")


    # extract layerwise data
    design = "ubrain"
    item = "BUF"
    area_ubr_h0_buf, power_ubr_h0_buf = layer_area_power_extract(h0_layers, design, item, workbook)
    area_ubr_ba_buf, power_ubr_ba_buf = layer_area_power_extract(ba_layers, design, item, workbook)
    area_ubr_bp_buf, power_ubr_bp_buf = layer_area_power_extract(bp_layers_ubr, design, item, workbook)
    item = "RNG"
    area_ubr_h0_rng, power_ubr_h0_rng = layer_area_power_extract(h0_layers, design, item, workbook)
    area_ubr_ba_rng, power_ubr_ba_rng = layer_area_power_extract(ba_layers, design, item, workbook)
    area_ubr_bp_rng, power_ubr_bp_rng = layer_area_power_extract(bp_layers_ubr, design, item, workbook)
    item = "CNT"
    area_ubr_h0_cnt, power_ubr_h0_cnt = layer_area_power_extract(h0_layers, design, item, workbook)
    area_ubr_ba_cnt, power_ubr_ba_cnt = layer_area_power_extract(ba_layers, design, item, workbook)
    area_ubr_bp_cnt, power_ubr_bp_cnt = layer_area_power_extract(bp_layers_ubr, design, item, workbook)
    item = "CMP"
    area_ubr_h0_cmp, power_ubr_h0_cmp = layer_area_power_extract(h0_layers, design, item, workbook)
    area_ubr_ba_cmp, power_ubr_ba_cmp = layer_area_power_extract(ba_layers, design, item, workbook)
    area_ubr_bp_cmp, power_ubr_bp_cmp = layer_area_power_extract(bp_layers_ubr, design, item, workbook)
    item = "PC"
    area_ubr_h0_pc, power_ubr_h0_pc = layer_area_power_extract(h0_layers, design, item, workbook)
    area_ubr_ba_pc, power_ubr_ba_pc = layer_area_power_extract(ba_layers, design, item, workbook)
    area_ubr_bp_pc, power_ubr_bp_pc = layer_area_power_extract(bp_layers_ubr, design, item, workbook)
    item = "REST"
    area_ubr_h0_rest, power_ubr_h0_rest = layer_area_power_extract(h0_layers, design, item, workbook)
    area_ubr_ba_rest, power_ubr_ba_rest = layer_area_power_extract(ba_layers, design, item, workbook)
    area_ubr_bp_rest, power_ubr_bp_rest = layer_area_power_extract(bp_layers_ubr, design, item, workbook)

    design = "sc"
    item = "BUF"
    area_sto_h0_buf, power_sto_h0_buf = layer_area_power_extract(h0_layers, design, item, workbook)
    area_sto_ba_buf, power_sto_ba_buf = layer_area_power_extract(ba_layers, design, item, workbook)
    area_sto_bp_buf, power_sto_bp_buf = layer_area_power_extract(bp_layers_sto, design, item, workbook)
    item = "RNG"
    area_sto_h0_rng, power_sto_h0_rng = layer_area_power_extract(h0_layers, design, item, workbook)
    area_sto_ba_rng, power_sto_ba_rng = layer_area_power_extract(ba_layers, design, item, workbook)
    area_sto_bp_rng, power_sto_bp_rng = layer_area_power_extract(bp_layers_sto, design, item, workbook)
    item = "CNT"
    area_sto_h0_cnt, power_sto_h0_cnt = layer_area_power_extract(h0_layers, design, item, workbook)
    area_sto_ba_cnt, power_sto_ba_cnt = layer_area_power_extract(ba_layers, design, item, workbook)
    area_sto_bp_cnt, power_sto_bp_cnt = layer_area_power_extract(bp_layers_sto, design, item, workbook)
    item = "CMP"
    area_sto_h0_cmp, power_sto_h0_cmp = layer_area_power_extract(h0_layers, design, item, workbook)
    area_sto_ba_cmp, power_sto_ba_cmp = layer_area_power_extract(ba_layers, design, item, workbook)
    area_sto_bp_cmp, power_sto_bp_cmp = layer_area_power_extract(bp_layers_sto, design, item, workbook)
    item = "PC"
    area_sto_h0_pc, power_sto_h0_pc = layer_area_power_extract(h0_layers, design, item, workbook)
    area_sto_ba_pc, power_sto_ba_pc = layer_area_power_extract(ba_layers, design, item, workbook)
    area_sto_bp_pc, power_sto_bp_pc = layer_area_power_extract(bp_layers_sto, design, item, workbook)
    item = "REST"
    area_sto_h0_rest, power_sto_h0_rest = layer_area_power_extract(h0_layers, design, item, workbook)
    area_sto_ba_rest, power_sto_ba_rest = layer_area_power_extract(ba_layers, design, item, workbook)
    area_sto_bp_rest, power_sto_bp_rest = layer_area_power_extract(bp_layers_sto, design, item, workbook)


    # area of layers in SC and uBrain
    my_dpi = 300
    fig_h = 3
    fig_w = 3.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    plt.ylabel("Area (mm$\mathregular{^2}$)")
    x_axis = ["SC", "SC-A", "SC-P", "uBrain", "uBrain-A", "uBrain-P"]
    x_idx = np.arange(len(x_axis))
    width = 0.5

    # conv1
    index = 0
    ax = plt.subplot(221)
    layer_buf = np.array([area_sto_h0_buf[index], area_sto_ba_buf[index], area_sto_bp_buf[index], area_ubr_h0_buf[index], area_ubr_ba_buf[index], area_ubr_bp_buf[index]])
    layer_rng = np.array([area_sto_h0_rng[index], area_sto_ba_rng[index], area_sto_bp_rng[index], area_ubr_h0_rng[index], area_ubr_ba_rng[index], area_ubr_bp_rng[index]])
    layer_cnt = np.array([area_sto_h0_cnt[index], area_sto_ba_cnt[index], area_sto_bp_cnt[index], area_ubr_h0_cnt[index], area_ubr_ba_cnt[index], area_ubr_bp_cnt[index]])
    layer_cmp = np.array([area_sto_h0_cmp[index], area_sto_ba_cmp[index], area_sto_bp_cmp[index], area_ubr_h0_cmp[index], area_ubr_ba_cmp[index], area_ubr_bp_cmp[index]])
    layer_pc = np.array([area_sto_h0_pc[index], area_sto_ba_pc[index], area_sto_bp_pc[index], area_ubr_h0_pc[index], area_ubr_ba_pc[index], area_ubr_bp_pc[index]])
    layer_rest = np.array([area_sto_h0_rest[index], area_sto_ba_rest[index], area_sto_bp_rest[index], area_ubr_h0_rest[index], area_ubr_ba_rest[index], area_ubr_bp_rest[index]])
    l1 = ax.bar(x_idx, layer_buf, width, hatch = None, alpha=0.99, color=gry, label='BUF')[0]
    current_bottom = layer_buf
    l2 = ax.bar(x_idx, layer_rng, width, bottom = current_bottom, hatch = None, alpha=0.99, color=sal, label='RNG')[0]
    current_bottom += layer_rng
    l3 = ax.bar(x_idx, layer_cnt, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr1, label='CNT')[0]
    current_bottom += layer_cnt
    l4 = ax.bar(x_idx, layer_cmp, width, bottom = current_bottom, hatch = None, alpha=0.99, color=orc, label='CMP')[0]
    current_bottom += layer_cmp
    l5 = ax.bar(x_idx, layer_pc, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr2, label='PC')[0]
    current_bottom += layer_pc
    l6 = ax.bar(x_idx, layer_rest, width, bottom = current_bottom, hatch = None, alpha=0.99, color=lav, label='REST')[0]
    current_bottom += layer_rest
    ax.set_title('Conv1')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_axis, rotation=30)

    # conv2
    index = 1
    ax = plt.subplot(222)
    layer_buf = np.array([area_sto_h0_buf[index], area_sto_ba_buf[index], area_sto_bp_buf[index], area_ubr_h0_buf[index], area_ubr_ba_buf[index], area_ubr_bp_buf[index]])
    layer_rng = np.array([area_sto_h0_rng[index], area_sto_ba_rng[index], area_sto_bp_rng[index], area_ubr_h0_rng[index], area_ubr_ba_rng[index], area_ubr_bp_rng[index]])
    layer_cnt = np.array([area_sto_h0_cnt[index], area_sto_ba_cnt[index], area_sto_bp_cnt[index], area_ubr_h0_cnt[index], area_ubr_ba_cnt[index], area_ubr_bp_cnt[index]])
    layer_cmp = np.array([area_sto_h0_cmp[index], area_sto_ba_cmp[index], area_sto_bp_cmp[index], area_ubr_h0_cmp[index], area_ubr_ba_cmp[index], area_ubr_bp_cmp[index]])
    layer_pc = np.array([area_sto_h0_pc[index], area_sto_ba_pc[index], area_sto_bp_pc[index], area_ubr_h0_pc[index], area_ubr_ba_pc[index], area_ubr_bp_pc[index]])
    layer_rest = np.array([area_sto_h0_rest[index], area_sto_ba_rest[index], area_sto_bp_rest[index], area_ubr_h0_rest[index], area_ubr_ba_rest[index], area_ubr_bp_rest[index]])
    ax.bar(x_idx, layer_buf, width, hatch = None, alpha=0.99, color=gry)
    current_bottom = layer_buf
    ax.bar(x_idx, layer_rng, width, bottom = current_bottom, hatch = None, alpha=0.99, color=sal)
    current_bottom += layer_rng
    ax.bar(x_idx, layer_cnt, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr1)
    current_bottom += layer_cnt
    ax.bar(x_idx, layer_cmp, width, bottom = current_bottom, hatch = None, alpha=0.99, color=orc)
    current_bottom += layer_cmp
    ax.bar(x_idx, layer_pc, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr2)
    current_bottom += layer_pc
    ax.bar(x_idx, layer_rest, width, bottom = current_bottom, hatch = None, alpha=0.99, color=lav)
    current_bottom += layer_rest
    ax.set_title('Conv2')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_axis, rotation=30)


    x_axis = ["SC", "uBrain"]
    x_idx = np.arange(len(x_axis))
    # fc3
    index = 2
    ax = plt.subplot(245)
    layer_buf = np.array([area_sto_h0_buf[index], area_ubr_h0_buf[index]])
    layer_rng = np.array([area_sto_h0_rng[index], area_ubr_h0_rng[index]])
    layer_cnt = np.array([area_sto_h0_cnt[index], area_ubr_h0_cnt[index]])
    layer_cmp = np.array([area_sto_h0_cmp[index], area_ubr_h0_cmp[index]])
    layer_pc = np.array([area_sto_h0_pc[index], area_ubr_h0_pc[index]])
    layer_rest = np.array([area_sto_h0_rest[index], area_ubr_h0_rest[index]])
    ax.bar(x_idx, layer_buf, width, hatch = None, alpha=0.99, color=gry)
    current_bottom = layer_buf
    ax.bar(x_idx, layer_rng, width, bottom = current_bottom, hatch = None, alpha=0.99, color=sal)
    current_bottom += layer_rng
    ax.bar(x_idx, layer_cnt, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr1)
    current_bottom += layer_cnt
    ax.bar(x_idx, layer_cmp, width, bottom = current_bottom, hatch = None, alpha=0.99, color=orc)
    current_bottom += layer_cmp
    ax.bar(x_idx, layer_pc, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr2)
    current_bottom += layer_pc
    ax.bar(x_idx, layer_rest, width, bottom = current_bottom, hatch = None, alpha=0.99, color=lav)
    current_bottom += layer_rest
    ax.set_title('FC3')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_axis, rotation=30)

    # mgu4
    index = 3
    ax = plt.subplot(246)
    layer_buf = np.array([area_sto_h0_buf[index], area_ubr_h0_buf[index]])
    layer_rng = np.array([area_sto_h0_rng[index], area_ubr_h0_rng[index]])
    layer_cnt = np.array([area_sto_h0_cnt[index], area_ubr_h0_cnt[index]])
    layer_cmp = np.array([area_sto_h0_cmp[index], area_ubr_h0_cmp[index]])
    layer_pc = np.array([area_sto_h0_pc[index], area_ubr_h0_pc[index]])
    layer_rest = np.array([area_sto_h0_rest[index], area_ubr_h0_rest[index]])
    ax.bar(x_idx, layer_buf, width, hatch = None, alpha=0.99, color=gry)
    current_bottom = layer_buf
    ax.bar(x_idx, layer_rng, width, bottom = current_bottom, hatch = None, alpha=0.99, color=sal)
    current_bottom += layer_rng
    ax.bar(x_idx, layer_cnt, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr1)
    current_bottom += layer_cnt
    ax.bar(x_idx, layer_cmp, width, bottom = current_bottom, hatch = None, alpha=0.99, color=orc)
    current_bottom += layer_cmp
    ax.bar(x_idx, layer_pc, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr2)
    current_bottom += layer_pc
    ax.bar(x_idx, layer_rest, width, bottom = current_bottom, hatch = None, alpha=0.99, color=lav)
    current_bottom += layer_rest
    ax.set_title('MGU4')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_axis, rotation=30)

    # fc5
    index = 4
    ax = plt.subplot(247)
    layer_buf = np.array([area_sto_h0_buf[index], area_ubr_h0_buf[index]])
    layer_rng = np.array([area_sto_h0_rng[index], area_ubr_h0_rng[index]])
    layer_cnt = np.array([area_sto_h0_cnt[index], area_ubr_h0_cnt[index]])
    layer_cmp = np.array([area_sto_h0_cmp[index], area_ubr_h0_cmp[index]])
    layer_pc = np.array([area_sto_h0_pc[index], area_ubr_h0_pc[index]])
    layer_rest = np.array([area_sto_h0_rest[index], area_ubr_h0_rest[index]])
    ax.bar(x_idx, layer_buf, width, hatch = None, alpha=0.99, color=gry)
    current_bottom = layer_buf
    ax.bar(x_idx, layer_rng, width, bottom = current_bottom, hatch = None, alpha=0.99, color=sal)
    current_bottom += layer_rng
    ax.bar(x_idx, layer_cnt, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr1)
    current_bottom += layer_cnt
    ax.bar(x_idx, layer_cmp, width, bottom = current_bottom, hatch = None, alpha=0.99, color=orc)
    current_bottom += layer_cmp
    ax.bar(x_idx, layer_pc, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr2)
    current_bottom += layer_pc
    ax.bar(x_idx, layer_rest, width, bottom = current_bottom, hatch = None, alpha=0.99, color=lav)
    current_bottom += layer_rest
    ax.set_title('FC5')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_axis, rotation=30)

    # fc6
    index = 5
    ax = plt.subplot(248)
    layer_buf = np.array([area_sto_h0_buf[index], area_ubr_h0_buf[index]])
    layer_rng = np.array([area_sto_h0_rng[index], area_ubr_h0_rng[index]])
    layer_cnt = np.array([area_sto_h0_cnt[index], area_ubr_h0_cnt[index]])
    layer_cmp = np.array([area_sto_h0_cmp[index], area_ubr_h0_cmp[index]])
    layer_pc = np.array([area_sto_h0_pc[index], area_ubr_h0_pc[index]])
    layer_rest = np.array([area_sto_h0_rest[index], area_ubr_h0_rest[index]])
    ax.bar(x_idx, layer_buf, width, hatch = None, alpha=0.99, color=gry)
    current_bottom = layer_buf
    ax.bar(x_idx, layer_rng, width, bottom = current_bottom, hatch = None, alpha=0.99, color=sal)
    current_bottom += layer_rng
    ax.bar(x_idx, layer_cnt, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr1)
    current_bottom += layer_cnt
    ax.bar(x_idx, layer_cmp, width, bottom = current_bottom, hatch = None, alpha=0.99, color=orc)
    current_bottom += layer_cmp
    ax.bar(x_idx, layer_pc, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr2)
    current_bottom += layer_pc
    ax.bar(x_idx, layer_rest, width, bottom = current_bottom, hatch = None, alpha=0.99, color=lav)
    current_bottom += layer_rest
    ax.set_title('FC6')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_axis, rotation=30)
    
    fig.tight_layout()
    fig.text(0.05, 0.5, 'Area (mm$\mathregular{^2}$)', ha='center', va='center', rotation='vertical')
    fig.subplots_adjust(top=0.85, left=0.15)
    line_labels = ["BUF", "RNG", "CNT", "CMP", "PC", "REST"]
    fig.legend([l1, l2, l3, l4, l5, l6], line_labels, loc="upper center", ncol=6, frameon=True)
    plt.savefig(output_path+"/Area_layerwise.pdf", bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)
    print("Layerwise area fig saved!\n")



    # power of layers in SC and uBrain
    my_dpi = 300
    fig_h = 3
    fig_w = 3.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    plt.ylabel("Area (mm$\mathregular{^2}$)")
    x_axis = ["SC", "SC-A", "SC-P", "uBrain", "uBrain-A", "uBrain-P"]
    x_idx = np.arange(len(x_axis))
    width = 0.5

    # conv1
    index = 0
    ax = plt.subplot(221)
    layer_buf = np.array([power_sto_h0_buf[index], power_sto_ba_buf[index], power_sto_bp_buf[index], power_ubr_h0_buf[index], power_ubr_ba_buf[index], power_ubr_bp_buf[index]])
    layer_rng = np.array([power_sto_h0_rng[index], power_sto_ba_rng[index], power_sto_bp_rng[index], power_ubr_h0_rng[index], power_ubr_ba_rng[index], power_ubr_bp_rng[index]])
    layer_cnt = np.array([power_sto_h0_cnt[index], power_sto_ba_cnt[index], power_sto_bp_cnt[index], power_ubr_h0_cnt[index], power_ubr_ba_cnt[index], power_ubr_bp_cnt[index]])
    layer_cmp = np.array([power_sto_h0_cmp[index], power_sto_ba_cmp[index], power_sto_bp_cmp[index], power_ubr_h0_cmp[index], power_ubr_ba_cmp[index], power_ubr_bp_cmp[index]])
    layer_pc = np.array([power_sto_h0_pc[index], power_sto_ba_pc[index], power_sto_bp_pc[index], power_ubr_h0_pc[index], power_ubr_ba_pc[index], power_ubr_bp_pc[index]])
    layer_rest = np.array([power_sto_h0_rest[index], power_sto_ba_rest[index], power_sto_bp_rest[index], power_ubr_h0_rest[index], power_ubr_ba_rest[index], power_ubr_bp_rest[index]])
    l1 = ax.bar(x_idx, layer_buf, width, hatch = None, alpha=0.99, color=gry, label='BUF')[0]
    current_bottom = layer_buf
    l2 = ax.bar(x_idx, layer_rng, width, bottom = current_bottom, hatch = None, alpha=0.99, color=sal, label='RNG')[0]
    current_bottom += layer_rng
    l3 = ax.bar(x_idx, layer_cnt, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr1, label='CNT')[0]
    current_bottom += layer_cnt
    l4 = ax.bar(x_idx, layer_cmp, width, bottom = current_bottom, hatch = None, alpha=0.99, color=orc, label='CMP')[0]
    current_bottom += layer_cmp
    l5 = ax.bar(x_idx, layer_pc, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr2, label='PC')[0]
    current_bottom += layer_pc
    l6 = ax.bar(x_idx, layer_rest, width, bottom = current_bottom, hatch = None, alpha=0.99, color=lav, label='REST')[0]
    current_bottom += layer_rest
    ax.set_title('Conv1')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_axis, rotation=30)

    # conv2
    index = 1
    ax = plt.subplot(222)
    layer_buf = np.array([power_sto_h0_buf[index], power_sto_ba_buf[index], power_sto_bp_buf[index], power_ubr_h0_buf[index], power_ubr_ba_buf[index], power_ubr_bp_buf[index]])
    layer_rng = np.array([power_sto_h0_rng[index], power_sto_ba_rng[index], power_sto_bp_rng[index], power_ubr_h0_rng[index], power_ubr_ba_rng[index], power_ubr_bp_rng[index]])
    layer_cnt = np.array([power_sto_h0_cnt[index], power_sto_ba_cnt[index], power_sto_bp_cnt[index], power_ubr_h0_cnt[index], power_ubr_ba_cnt[index], power_ubr_bp_cnt[index]])
    layer_cmp = np.array([power_sto_h0_cmp[index], power_sto_ba_cmp[index], power_sto_bp_cmp[index], power_ubr_h0_cmp[index], power_ubr_ba_cmp[index], power_ubr_bp_cmp[index]])
    layer_pc = np.array([power_sto_h0_pc[index], power_sto_ba_pc[index], power_sto_bp_pc[index], power_ubr_h0_pc[index], power_ubr_ba_pc[index], power_ubr_bp_pc[index]])
    layer_rest = np.array([power_sto_h0_rest[index], power_sto_ba_rest[index], power_sto_bp_rest[index], power_ubr_h0_rest[index], power_ubr_ba_rest[index], power_ubr_bp_rest[index]])
    ax.bar(x_idx, layer_buf, width, hatch = None, alpha=0.99, color=gry)
    current_bottom = layer_buf
    ax.bar(x_idx, layer_rng, width, bottom = current_bottom, hatch = None, alpha=0.99, color=sal)
    current_bottom += layer_rng
    ax.bar(x_idx, layer_cnt, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr1)
    current_bottom += layer_cnt
    ax.bar(x_idx, layer_cmp, width, bottom = current_bottom, hatch = None, alpha=0.99, color=orc)
    current_bottom += layer_cmp
    ax.bar(x_idx, layer_pc, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr2)
    current_bottom += layer_pc
    ax.bar(x_idx, layer_rest, width, bottom = current_bottom, hatch = None, alpha=0.99, color=lav)
    current_bottom += layer_rest
    ax.set_title('Conv2')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_axis, rotation=30)


    x_axis = ["SC", "uBrain"]
    x_idx = np.arange(len(x_axis))
    # fc3
    index = 2
    ax = plt.subplot(245)
    layer_buf = np.array([power_sto_h0_buf[index], power_ubr_h0_buf[index]])
    layer_rng = np.array([power_sto_h0_rng[index], power_ubr_h0_rng[index]])
    layer_cnt = np.array([power_sto_h0_cnt[index], power_ubr_h0_cnt[index]])
    layer_cmp = np.array([power_sto_h0_cmp[index], power_ubr_h0_cmp[index]])
    layer_pc = np.array([power_sto_h0_pc[index], power_ubr_h0_pc[index]])
    layer_rest = np.array([power_sto_h0_rest[index], power_ubr_h0_rest[index]])
    ax.bar(x_idx, layer_buf, width, hatch = None, alpha=0.99, color=gry)
    current_bottom = layer_buf
    ax.bar(x_idx, layer_rng, width, bottom = current_bottom, hatch = None, alpha=0.99, color=sal)
    current_bottom += layer_rng
    ax.bar(x_idx, layer_cnt, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr1)
    current_bottom += layer_cnt
    ax.bar(x_idx, layer_cmp, width, bottom = current_bottom, hatch = None, alpha=0.99, color=orc)
    current_bottom += layer_cmp
    ax.bar(x_idx, layer_pc, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr2)
    current_bottom += layer_pc
    ax.bar(x_idx, layer_rest, width, bottom = current_bottom, hatch = None, alpha=0.99, color=lav)
    current_bottom += layer_rest
    ax.set_title('FC3')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_axis, rotation=30)

    # mgu4
    index = 3
    ax = plt.subplot(246)
    layer_buf = np.array([power_sto_h0_buf[index], power_ubr_h0_buf[index]])
    layer_rng = np.array([power_sto_h0_rng[index], power_ubr_h0_rng[index]])
    layer_cnt = np.array([power_sto_h0_cnt[index], power_ubr_h0_cnt[index]])
    layer_cmp = np.array([power_sto_h0_cmp[index], power_ubr_h0_cmp[index]])
    layer_pc = np.array([power_sto_h0_pc[index], power_ubr_h0_pc[index]])
    layer_rest = np.array([power_sto_h0_rest[index], power_ubr_h0_rest[index]])
    ax.bar(x_idx, layer_buf, width, hatch = None, alpha=0.99, color=gry)
    current_bottom = layer_buf
    ax.bar(x_idx, layer_rng, width, bottom = current_bottom, hatch = None, alpha=0.99, color=sal)
    current_bottom += layer_rng
    ax.bar(x_idx, layer_cnt, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr1)
    current_bottom += layer_cnt
    ax.bar(x_idx, layer_cmp, width, bottom = current_bottom, hatch = None, alpha=0.99, color=orc)
    current_bottom += layer_cmp
    ax.bar(x_idx, layer_pc, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr2)
    current_bottom += layer_pc
    ax.bar(x_idx, layer_rest, width, bottom = current_bottom, hatch = None, alpha=0.99, color=lav)
    current_bottom += layer_rest
    ax.set_title('MGU4')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_axis, rotation=30)

    # fc5
    index = 4
    ax = plt.subplot(247)
    layer_buf = np.array([power_sto_h0_buf[index], power_ubr_h0_buf[index]])
    layer_rng = np.array([power_sto_h0_rng[index], power_ubr_h0_rng[index]])
    layer_cnt = np.array([power_sto_h0_cnt[index], power_ubr_h0_cnt[index]])
    layer_cmp = np.array([power_sto_h0_cmp[index], power_ubr_h0_cmp[index]])
    layer_pc = np.array([power_sto_h0_pc[index], power_ubr_h0_pc[index]])
    layer_rest = np.array([power_sto_h0_rest[index], power_ubr_h0_rest[index]])
    ax.bar(x_idx, layer_buf, width, hatch = None, alpha=0.99, color=gry)
    current_bottom = layer_buf
    ax.bar(x_idx, layer_rng, width, bottom = current_bottom, hatch = None, alpha=0.99, color=sal)
    current_bottom += layer_rng
    ax.bar(x_idx, layer_cnt, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr1)
    current_bottom += layer_cnt
    ax.bar(x_idx, layer_cmp, width, bottom = current_bottom, hatch = None, alpha=0.99, color=orc)
    current_bottom += layer_cmp
    ax.bar(x_idx, layer_pc, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr2)
    current_bottom += layer_pc
    ax.bar(x_idx, layer_rest, width, bottom = current_bottom, hatch = None, alpha=0.99, color=lav)
    current_bottom += layer_rest
    ax.set_title('FC5')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_axis, rotation=30)

    # fc6
    index = 5
    ax = plt.subplot(248)
    layer_buf = np.array([power_sto_h0_buf[index], power_ubr_h0_buf[index]])
    layer_rng = np.array([power_sto_h0_rng[index], power_ubr_h0_rng[index]])
    layer_cnt = np.array([power_sto_h0_cnt[index], power_ubr_h0_cnt[index]])
    layer_cmp = np.array([power_sto_h0_cmp[index], power_ubr_h0_cmp[index]])
    layer_pc = np.array([power_sto_h0_pc[index], power_ubr_h0_pc[index]])
    layer_rest = np.array([power_sto_h0_rest[index], power_ubr_h0_rest[index]])
    ax.bar(x_idx, layer_buf, width, hatch = None, alpha=0.99, color=gry)
    current_bottom = layer_buf
    ax.bar(x_idx, layer_rng, width, bottom = current_bottom, hatch = None, alpha=0.99, color=sal)
    current_bottom += layer_rng
    ax.bar(x_idx, layer_cnt, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr1)
    current_bottom += layer_cnt
    ax.bar(x_idx, layer_cmp, width, bottom = current_bottom, hatch = None, alpha=0.99, color=orc)
    current_bottom += layer_cmp
    ax.bar(x_idx, layer_pc, width, bottom = current_bottom, hatch = None, alpha=0.99, color=gr2)
    current_bottom += layer_pc
    ax.bar(x_idx, layer_rest, width, bottom = current_bottom, hatch = None, alpha=0.99, color=lav)
    current_bottom += layer_rest
    ax.set_title('FC6')
    ax.set_xticks(x_idx)
    ax.set_xticklabels(x_axis, rotation=30)
    
    fig.tight_layout()
    fig.text(0.05, 0.5, 'Power (mW)', ha='center', va='center', rotation='vertical')
    fig.subplots_adjust(top=0.85, left=0.15)
    line_labels = ["BUF", "RNG", "CNT", "CMP", "PC", "REST"]
    fig.legend([l1, l2, l3, l4, l5, l6], line_labels, loc="upper center", ncol=6, frameon=True)
    plt.savefig(output_path+"/Power_layerwise.pdf", bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)
    print("Layerwise power fig saved!\n")

    
    print("Sense area improvement (all channels): {:3.2f}".format(area_sense_bl/area_sense_ubr))
    print()
    print("Sense power improvement (all channels): {:3.2f}".format(power_sense_bl/power_sense_ubr))
    print()

    print("Compute (On-chip) area improvement:")
    best_cpu_onchip_area = area_cpu[-1]
    best_sys_onchip_area = area_sys[-1]
    best_sto_onchip_area = area_sto_ba_onchip
    best_ubr_onchip_area = area_ubr_ba_onchip
    print("\tOver CPU           : {:3.2f}".format(best_cpu_onchip_area / best_ubr_onchip_area))
    print("\tOver Systolic array: {:3.2f}".format(best_sys_onchip_area / best_ubr_onchip_area))
    print("\tOver Stochastic    : {:3.2f}".format(best_sto_onchip_area / best_ubr_onchip_area))
    print()

    print("Compute (On-chip) power improvement:")
    best_cpu_onchip_power = best_onchip_power_list[0]
    best_sys_onchip_power = best_onchip_power_list[1]
    best_sto_onchip_power = best_onchip_power_list[4]
    best_ubr_onchip_power = best_onchip_power_list[7]
    print("\tOver CPU           : {:3.2f}".format(best_cpu_onchip_power / best_ubr_onchip_power))
    print("\tOver Systolic array: {:3.2f}".format(best_sys_onchip_power / best_ubr_onchip_power))
    print("\tOver Stochastic    : {:3.2f}".format(best_sto_onchip_power / best_ubr_onchip_power))
    print()

    # get max improve
    design = "ubrain"
    item = "TOTAL"
    area_ubr_h0_total, power_ubr_h0_total = layer_area_power_extract(h0_layers, design, item, workbook)
    area_ubr_ba_total, power_ubr_ba_total = layer_area_power_extract(ba_layers, design, item, workbook)
    area_ubr_bp_total, power_ubr_bp_total = layer_area_power_extract(bp_layers_ubr, design, item, workbook)

    design = "sc"
    item = "TOTAL"
    area_sto_h0_total, power_sto_h0_total = layer_area_power_extract(h0_layers, design, item, workbook)
    area_sto_ba_total, power_sto_ba_total = layer_area_power_extract(ba_layers, design, item, workbook)
    area_sto_bp_total, power_sto_bp_total = layer_area_power_extract(bp_layers_sto, design, item, workbook)

    print("Max layerwise area improve:")
    print("\tOver SC        : {:3.2f} at index {:2d}".format(np.amax(np.array(area_sto_h0_total) / np.array(area_ubr_ba_total)), 
                                                            np.argmax(np.array(area_sto_h0_total) / np.array(area_ubr_ba_total))))
    print("\tOver SC-A      : {:3.2f} at index {:2d}".format(np.amax(np.array(area_sto_ba_total) / np.array(area_ubr_ba_total)), 
                                                            np.argmax(np.array(area_sto_ba_total) / np.array(area_ubr_ba_total))))
    print("\tOver SC-P      : {:3.2f} at index {:2d}".format(np.amax(np.array(area_sto_bp_total) / np.array(area_ubr_ba_total)), 
                                                            np.argmax(np.array(area_sto_bp_total) / np.array(area_ubr_ba_total))))
    print()

    print("Max layerwise power improve:")
    print("\tOver SC        : {:3.2f} at index {:2d}".format(np.amax(np.array(power_sto_h0_total) / np.array(power_ubr_bp_total)), 
                                                            np.argmax(np.array(power_sto_h0_total) / np.array(power_ubr_bp_total))))
    print("\tOver SC-A      : {:3.2f} at index {:2d}".format(np.amax(np.array(power_sto_ba_total) / np.array(power_ubr_bp_total)), 
                                                            np.argmax(np.array(power_sto_ba_total) / np.array(power_ubr_bp_total))))
    print("\tOver SC-P      : {:3.2f} at index {:2d}".format(np.amax(np.array(power_sto_bp_total) / np.array(power_ubr_bp_total)), 
                                                            np.argmax(np.array(power_sto_bp_total) / np.array(power_ubr_bp_total))))
    print()

    print("Max iso-latency power improve:")
    closest_runtime_idx_cpu = np.argmin(np.abs(np.array(runtime_cpu) + runtime_window - runtime_unary))
    closest_runtime_idx_sys = np.argmin(np.abs(np.array(runtime_sys) + runtime_window - runtime_unary))
    # print("CPU runtime              : ", np.array(runtime_cpu) + runtime_window)
    # print("Closest CPU runtime      : ", runtime_cpu[closest_runtime_idx_cpu] + runtime_window)
    # print("Systolic runtime         : ", np.array(runtime_sys) + runtime_window)
    # print("Closest Systolic runtime : ", runtime_sys[closest_runtime_idx_sys] + runtime_window)
    print("\tOver CPU       : {:3.2f}".format(np.amax(np.array(np.sum(power_cpu[closest_runtime_idx_cpu])) / np.sum(power_ubr_bp)))) 
    print("\tOver Systolic  : {:3.2f}".format(np.amax(np.array(np.sum(power_sys[closest_runtime_idx_sys])) / np.sum(power_ubr_bp))))
    print()

    print("Max iso-latency frequency decrease:")
    ubr_sto_max_freq = 4.2
    print("\tOver CPU       : {:3.2f}".format(np.amax(np.array(freq_cpu[closest_runtime_idx_cpu]) / ubr_sto_max_freq))) 
    print("\tOver Systolic  : {:3.2f}".format(np.amax(np.array(freq_sys[closest_runtime_idx_sys]) / ubr_sto_max_freq)))
    print()
    

def layer_area_power_extract(layers=[], design="sc", item="", workbook=None):
    index = ["BUF", "RNG", "CNT", "CMP", "PC", "REST", "TOTAL"].index(item)
    area = []
    area.append(sc_ubrain_report(workbook, layers[0], design)[0][index])
    area.append(sc_ubrain_report(workbook, layers[1], design)[0][index])
    area.append(sc_ubrain_report(workbook, layers[2], design)[0][index])
    area.append(sc_ubrain_report(workbook, layers[3], design)[0][index])
    area.append(sc_ubrain_report(workbook, layers[4], design)[0][index])
    area.append(sc_ubrain_report(workbook, layers[5], design)[0][index])
    power = []
    power.append(sc_ubrain_report(workbook, layers[0], design)[3][index])
    power.append(sc_ubrain_report(workbook, layers[1], design)[3][index])
    power.append(sc_ubrain_report(workbook, layers[2], design)[3][index])
    power.append(sc_ubrain_report(workbook, layers[3], design)[3][index])
    power.append(sc_ubrain_report(workbook, layers[4], design)[3][index])
    power.append(sc_ubrain_report(workbook, layers[5], design)[3][index])
    return area, power


def design_area_power_extract(layers=[], design="sc", item="", workbook=None):
    index = ["BUF", "RNG", "CNT", "CMP", "PC", "REST", "TOTAL"].index(item)
    area = 0
    area += sc_ubrain_report(workbook, layers[0], design)[0][index]
    area += sc_ubrain_report(workbook, layers[1], design)[0][index]
    area += sc_ubrain_report(workbook, layers[2], design)[0][index]
    area += sc_ubrain_report(workbook, layers[3], design)[0][index]
    area += sc_ubrain_report(workbook, layers[4], design)[0][index]
    area += sc_ubrain_report(workbook, layers[5], design)[0][index]
    power = 0
    power += sc_ubrain_report(workbook, layers[0], design)[3][index]
    power += sc_ubrain_report(workbook, layers[1], design)[3][index]
    power += sc_ubrain_report(workbook, layers[2], design)[3][index]
    power += sc_ubrain_report(workbook, layers[3], design)[3][index]
    power += sc_ubrain_report(workbook, layers[4], design)[3][index]
    power += sc_ubrain_report(workbook, layers[5], design)[3][index]
    return area, power


if __name__ == "__main__":
    bci_hw_report()