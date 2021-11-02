#! /usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from UnarySim.kernel.rnn import HUBMGUCell, HardMGUCell, HardMGUCellFxp
from UnarySim.stream.gen import *
from UnarySim.metric.metric import SourceGen, RNG, BSGen, ProgError
from UnarySim.kernel.utils import truncated_normal, progerror_report
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bitwidth_list = [8, 9, 10, 11, 12]
output_dir = "/home/diwu/Project/UnarySim/app/uBrain/layer_eval/"

win_sz = 10 # win size
batch = 32
input_sz = 256 # fc size
hidden_sz = 64 # hidden size
intwidth = 1 # systolic array fxp
mode = "bipolar"
rng = "Sobol"
bias = False

err_array = np.zeros((len(bitwidth_list), 2, win_sz))

outfile = "layer_eval_rc_log.csv"
fp = open(output_dir+outfile, "w")

for bitwidth_index in range(len(bitwidth_list)):
    bitwidth = bitwidth_list[bitwidth_index]
    print("bit width:", bitwidth)

    fracwidth = bitwidth - intwidth
    depth = bitwidth + 2
    depth_ismul = bitwidth - 4

    input = torch.randn(win_sz, batch, input_sz).to(device)
    input = truncated_normal(input, mean=0, std=0.4)
    hx1 = torch.randn(batch, hidden_sz).to(device)
    hx1 = truncated_normal(hx1, mean=0, std=0.1)
    hx2 = hx1.clone().detach().to(device)
    hx3 = hx1.clone().detach().to(device)
    output1 = []
    output2 = []
    output3 = []

    rnn1 = HardMGUCell(input_sz, hidden_sz, bias=bias, hard=True).to(device)

    rnn2 = HUBMGUCell(input_sz, hidden_sz, bias=bias, 
                    binary_weight_f=rnn1.weight_f, binary_bias_f=rnn1.bias_f, binary_weight_n=rnn1.weight_n, binary_bias_n=rnn1.bias_n, 
                    rng=rng, bitwidth=bitwidth, mode=mode, depth=depth, depth_ismul=depth_ismul).to(device)

    rnn3 = HardMGUCellFxp(input_sz, hidden_sz, bias=bias, hard=True, intwidth=intwidth, fracwidth=fracwidth).to(device)
    rnn3.weight_f.data = rnn1.weight_f.clone().detach().to(device)
    rnn3.weight_n.data = rnn1.weight_n.clone().detach().to(device)

    for i in range(win_sz):
        hx1 = rnn1(input[i], hx1)
        output1.append(hx1)

        hx2 = rnn2(input[i], hx2)
        output2.append(hx2)

        hx3 = rnn3(input[i], hx3)
        output3.append(hx3)

        hub_err = hx1 - hx2
        min = hub_err.min().item()
        max = hub_err.max().item()
        rmse = torch.sqrt(torch.mean(torch.square(hub_err)))
        std, mean = torch.std_mean(hub_err)
        log = "{:30s}".format(str(i)+"-th win output hub") + \
                ", Absolute Error range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
                ", std," + "{:12f}".format(std) + \
                ", mean," + "{:12f}".format(mean) + \
                ", rmse," + "{:12f}".format(rmse)
        print(log)
        fp.write(log+"\n")
        err_array[bitwidth_index, 0, i] = rmse.cpu().item()

        fxp_err = hx1 - hx3
        min = fxp_err.min().item()
        max = fxp_err.max().item()
        rmse = torch.sqrt(torch.mean(torch.square(fxp_err)))
        std, mean = torch.std_mean(fxp_err)
        log = "{:30s}".format(str(i)+"-th win output fxp") + \
                ", Absolute Error range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
                ", std," + "{:12f}".format(std) + \
                ", mean," + "{:12f}".format(mean) + \
                ", rmse," + "{:12f}".format(rmse)
        print(log)
        fp.write(log+"\n")
        err_array[bitwidth_index, 1, i] = rmse.cpu().item()
    print()

print(err_array)
fp.write(str(err_array)+"\n")
fp.close()

font = {'family':'Times New Roman', 'size': 6}
matplotlib.rc('font', **font)

my_dpi = 300
fig_h = 0.8
fig_w = 3.6
alpha = 1

labels = [str(bitwidth) for bitwidth in bitwidth_list]
x = np.arange(len(labels))  # the label locations

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=my_dpi)

for idx in range(len(bitwidth_list)):
    data_hub = err_array[idx, 0, :]
    data_fxp = err_array[idx, 1, :]
    interval = 1/2**(win_sz.bit_length())
    x_axe = [(x[idx] - (win_sz - 1) / 2 * interval + x_tick * interval) for x_tick in range(win_sz)]
    if idx == 0:
        ax.plot(x_axe, data_hub, "-s", label="RC", alpha=alpha, color="#FF7F7F", lw=0.5, ms=1)
        ax.plot(x_axe, data_fxp, "-^", label="FXP", alpha=alpha, color="#7A81FF", lw=0.5, ms=1)
    else:
        ax.plot(x_axe, data_hub, "-s", alpha=alpha, color="#FF7F7F", lw=0.5, ms=1)
        ax.plot(x_axe, data_fxp, "-^", alpha=alpha, color="#7A81FF", lw=0.5, ms=1)


locs = [0, 0.05, 0.1]
ax.set_yticks(locs)
y_label_list = []
for y in locs:
    if y != 0:
        y_label_list.append("{:1.0E}".format(abs(y)))
    else:
        y_label_list.append("0")
ax.set_yticklabels(y_label_list)


ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('RMSE\n')
ax.legend(ncol=2, frameon=True)
fig.tight_layout()
fig.savefig(output_dir+"layer_eval_rc.pdf", bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)
