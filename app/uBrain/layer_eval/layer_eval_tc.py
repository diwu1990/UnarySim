#! /usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from UnarySim.kernel.linear import HUBLinear
from UnarySim.kernel.conv import HUBConv2d
from UnarySim.stream.gen import *
from UnarySim.metric.metric import SourceGen, RNG, BSGen, ProgError
from UnarySim.kernel.utils import truncated_normal, progerror_report, Round
from UnarySim.kernel.relu import ScaleReLU
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bitwidth_list = [8, 9, 10, 11, 12]
output_dir = "/home/diwu/Project/UnarySim/app/uBrain/layer_eval/"

batch = 32
cnn_chn = 16
cnn_kn_sz = 3
cnn_padding = 1
bias = False
rng = "Race"
input_sz = [10, 11]
fc_sz = 256
rnn_hidden_sz = 64
num_class = [5, 2]

intwidth = 1 # systolic array fxp

mode = "bipolar"

outfile = "layer_eval_tc_log.csv"
fp = open(output_dir+outfile, "w")

err_array = np.zeros((len(bitwidth_list), 2, 5))

for bitwidth_index in range(len(bitwidth_list)):
    bitwidth = bitwidth_list[bitwidth_index]
    cycle = 2**bitwidth
    print("bit width:", bitwidth)

    fracwidth = bitwidth - intwidth
    depth = bitwidth + 2
    depth_ismul = bitwidth - 4

    conv1_input = torch.randn(batch, 1, input_sz[0], input_sz[1]).to(device)
    conv1_input = truncated_normal(conv1_input, mean=0, std=0.5)
    conv2_input = torch.randn(batch, cnn_chn, input_sz[0], input_sz[1]).to(device)
    conv2_input = truncated_normal(conv2_input, mean=0, std=0.5)
    fc3_input = torch.randn(batch, (input_sz[0]+2*2*(cnn_padding-1))*(input_sz[1]+2*2*(cnn_padding-1))*cnn_chn*2).to(device)
    fc3_input = truncated_normal(fc3_input, mean=0, std=0.5)
    fc5_input = torch.randn(batch, rnn_hidden_sz).to(device)
    fc5_input = truncated_normal(fc5_input, mean=0, std=0.5)
    fc6_input = torch.randn(batch, rnn_hidden_sz).to(device)
    fc6_input = truncated_normal(fc6_input, mean=0, std=0.5)

    trunc = Round(intwidth=intwidth, fracwidth=fracwidth)

    conv1                   = nn.Conv2d(1        , cnn_chn  , (cnn_kn_sz, cnn_kn_sz), bias=bias, padding=cnn_padding).to(device)
    conv1.weight.data       = truncated_normal(conv1.weight, mean=0, std=0.5)
    conv2                   = nn.Conv2d(cnn_chn  , cnn_chn*2, (cnn_kn_sz, cnn_kn_sz), bias=bias, padding=cnn_padding).to(device)
    conv2.weight.data       = truncated_normal(conv2.weight, mean=0, std=0.5)
    fc3                     = nn.Linear((input_sz[0]+2*2*(cnn_padding-1))*(input_sz[1]+2*2*(cnn_padding-1))*cnn_chn*2, fc_sz, bias=bias).to(device)
    fc3.weight.data         = truncated_normal(fc3.weight, mean=0, std=0.5)
    fc5                     = nn.Linear(rnn_hidden_sz, num_class[0], bias=bias).to(device)
    fc5.weight.data         = truncated_normal(fc5.weight, mean=0, std=0.5)
    fc6                     = nn.Linear(rnn_hidden_sz, num_class[1], bias=bias).to(device)
    fc6.weight.data         = truncated_normal(fc6.weight, mean=0, std=0.5)

    conv1_fxp               = nn.Conv2d(1        , cnn_chn  , (cnn_kn_sz, cnn_kn_sz), bias=bias, padding=cnn_padding).to(device)
    conv1_fxp.weight.data   = trunc(conv1.weight)
    conv2_fxp               = nn.Conv2d(cnn_chn  , cnn_chn*2, (cnn_kn_sz, cnn_kn_sz), bias=bias, padding=cnn_padding).to(device)
    conv2_fxp.weight.data   = trunc(conv2.weight)
    fc3_fxp                 = nn.Linear((input_sz[0]+2*2*(cnn_padding-1))*(input_sz[1]+2*2*(cnn_padding-1))*cnn_chn*2, fc_sz, bias=bias).to(device)
    fc3_fxp.weight.data     = trunc(fc3.weight)
    fc5_fxp                 = nn.Linear(rnn_hidden_sz, num_class[0], bias=bias).to(device)
    fc5_fxp.weight.data     = trunc(fc5.weight)
    fc6_fxp                 = nn.Linear(rnn_hidden_sz, num_class[1], bias=bias).to(device)
    fc6_fxp.weight.data     = trunc(fc6.weight)

    conv1_hub               = HUBConv2d(1        , cnn_chn  , (cnn_kn_sz, cnn_kn_sz), bias=bias, padding=cnn_padding, 
                                    binary_weight=conv1.weight, binary_bias=conv1.bias, rng=rng, cycle=cycle).to(device)
    conv2_hub               = HUBConv2d(cnn_chn  , cnn_chn*2, (cnn_kn_sz, cnn_kn_sz), bias=bias, padding=cnn_padding, 
                                    binary_weight=conv2.weight, binary_bias=conv2.bias, rng=rng, cycle=cycle).to(device)
    fc3_hub                 = HUBLinear((input_sz[0]+2*2*(cnn_padding-1))*(input_sz[1]+2*2*(cnn_padding-1))*cnn_chn*2, fc_sz, bias=bias, 
                                    binary_weight=fc3.weight, binary_bias=fc3.bias, rng=rng, cycle=cycle).to(device)
    fc5_hub                 = HUBLinear(rnn_hidden_sz, num_class[0], bias=bias, 
                                    binary_weight=fc5.weight, binary_bias=fc5.bias, rng=rng, cycle=cycle).to(device)
    fc6_hub                 = HUBLinear(rnn_hidden_sz, num_class[1], bias=bias, 
                                    binary_weight=fc6.weight, binary_bias=fc6.bias, rng=rng, cycle=cycle).to(device)

    conv1_o         = ScaleReLU()(conv1(conv1_input.clone().detach()))
    conv2_o         = ScaleReLU()(conv2(conv2_input.clone().detach()))
    fc3_o           = ScaleReLU()(fc3(fc3_input.clone().detach()))
    fc5_o           = nn.Hardtanh()(fc5(fc5_input.clone().detach()))
    fc6_o           = nn.Hardtanh()(fc6(fc6_input.clone().detach()))

    conv1_o_fxp     = ScaleReLU()(trunc(conv1_fxp(trunc(conv1_input.clone().detach()))))
    conv2_o_fxp     = ScaleReLU()(trunc(conv2_fxp(trunc(conv2_input.clone().detach()))))
    fc3_o_fxp       = ScaleReLU()(trunc(fc3_fxp(trunc(fc3_input.clone().detach()))))
    fc5_o_fxp       = nn.Hardtanh()(trunc(fc5_fxp(trunc(fc5_input.clone().detach()))))
    fc6_o_fxp       = nn.Hardtanh()(trunc(fc6_fxp(trunc(fc6_input.clone().detach()))))

    conv1_o_hub     = ScaleReLU()(conv1_hub(conv1_input.clone().detach()))
    conv2_o_hub     = ScaleReLU()(conv2_hub(conv2_input.clone().detach()))
    fc3_o_hub       = ScaleReLU()(fc3_hub(fc3_input.clone().detach()))
    fc5_o_hub       = nn.Hardtanh()(fc5_hub(fc5_input.clone().detach()))
    fc6_o_hub       = nn.Hardtanh()(fc6_hub(fc6_input.clone().detach()))


    conv1_o_hub_err = conv1_o - conv1_o_hub
    min = conv1_o_hub_err.min().item()
    max = conv1_o_hub_err.max().item()
    rmse = torch.sqrt(torch.mean(torch.square(conv1_o_hub_err)))
    std, mean = torch.std_mean(conv1_o_hub_err)
    log = "{:30s}".format("conv1 hub") + \
            ", Absolute Error range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
            ", std," + "{:12f}".format(std) + \
            ", mean," + "{:12f}".format(mean) + \
            ", rmse," + "{:12f}".format(rmse)
    print(log)
    fp.write(log+"\n")
    err_array[bitwidth_index, 0, 0] = rmse.cpu().item()

    conv2_o_hub_err = conv2_o - conv2_o_hub
    min = conv2_o_hub_err.min().item()
    max = conv2_o_hub_err.max().item()
    rmse = torch.sqrt(torch.mean(torch.square(conv2_o_hub_err)))
    std, mean = torch.std_mean(conv2_o_hub_err)
    log = "{:30s}".format("conv2 hub") + \
            ", Absolute Error range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
            ", std," + "{:12f}".format(std) + \
            ", mean," + "{:12f}".format(mean) + \
            ", rmse," + "{:12f}".format(rmse)
    print(log)
    fp.write(log+"\n")
    err_array[bitwidth_index, 0, 1] = rmse.cpu().item()

    fc3_o_hub_err = fc3_o - fc3_o_hub
    min = fc3_o_hub_err.min().item()
    max = fc3_o_hub_err.max().item()
    rmse = torch.sqrt(torch.mean(torch.square(fc3_o_hub_err)))
    std, mean = torch.std_mean(fc3_o_hub_err)
    log = "{:30s}".format("fc3 hub") + \
            ", Absolute Error range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
            ", std," + "{:12f}".format(std) + \
            ", mean," + "{:12f}".format(mean) + \
            ", rmse," + "{:12f}".format(rmse)
    print(log)
    fp.write(log+"\n")
    err_array[bitwidth_index, 0, 2] = rmse.cpu().item()

    fc5_o_hub_err = fc5_o - fc5_o_hub
    min = fc5_o_hub_err.min().item()
    max = fc5_o_hub_err.max().item()
    rmse = torch.sqrt(torch.mean(torch.square(fc5_o_hub_err)))
    std, mean = torch.std_mean(fc5_o_hub_err)
    log = "{:30s}".format("fc5 hub") + \
            ", Absolute Error range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
            ", std," + "{:12f}".format(std) + \
            ", mean," + "{:12f}".format(mean) + \
            ", rmse," + "{:12f}".format(rmse)
    print(log)
    fp.write(log+"\n")
    err_array[bitwidth_index, 0, 3] = rmse.cpu().item()

    fc6_o_hub_err = fc6_o - fc6_o_hub
    min = fc6_o_hub_err.min().item()
    max = fc6_o_hub_err.max().item()
    rmse = torch.sqrt(torch.mean(torch.square(fc6_o_hub_err)))
    std, mean = torch.std_mean(fc6_o_hub_err)
    log = "{:30s}".format("fc6 hub") + \
            ", Absolute Error range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
            ", std," + "{:12f}".format(std) + \
            ", mean," + "{:12f}".format(mean) + \
            ", rmse," + "{:12f}".format(rmse)
    print(log)
    fp.write(log+"\n")
    err_array[bitwidth_index, 0, 4] = rmse.cpu().item()

    conv1_o_fxp_err = conv1_o - conv1_o_fxp
    min = conv1_o_fxp_err.min().item()
    max = conv1_o_fxp_err.max().item()
    rmse = torch.sqrt(torch.mean(torch.square(conv1_o_fxp_err)))
    std, mean = torch.std_mean(conv1_o_fxp_err)
    log = "{:30s}".format("conv1 fxp") + \
            ", Absolute Error range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
            ", std," + "{:12f}".format(std) + \
            ", mean," + "{:12f}".format(mean) + \
            ", rmse," + "{:12f}".format(rmse)
    print(log)
    fp.write(log+"\n")
    err_array[bitwidth_index, 1, 0] = rmse.cpu().item()

    conv2_o_fxp_err = conv2_o - conv2_o_fxp
    min = conv2_o_fxp_err.min().item()
    max = conv2_o_fxp_err.max().item()
    rmse = torch.sqrt(torch.mean(torch.square(conv2_o_fxp_err)))
    std, mean = torch.std_mean(conv2_o_fxp_err)
    log = "{:30s}".format("conv2 fxp") + \
            ", Absolute Error range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
            ", std," + "{:12f}".format(std) + \
            ", mean," + "{:12f}".format(mean) + \
            ", rmse," + "{:12f}".format(rmse)
    print(log)
    fp.write(log+"\n")
    err_array[bitwidth_index, 1, 1] = rmse.cpu().item()

    fc3_o_fxp_err = fc3_o - fc3_o_fxp
    min = fc3_o_fxp_err.min().item()
    max = fc3_o_fxp_err.max().item()
    rmse = torch.sqrt(torch.mean(torch.square(fc3_o_fxp_err)))
    std, mean = torch.std_mean(fc3_o_fxp_err)
    log = "{:30s}".format("fc3 fxp") + \
            ", Absolute Error range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
            ", std," + "{:12f}".format(std) + \
            ", mean," + "{:12f}".format(mean) + \
            ", rmse," + "{:12f}".format(rmse)
    print(log)
    fp.write(log+"\n")
    err_array[bitwidth_index, 1, 2] = rmse.cpu().item()

    fc5_o_fxp_err = fc5_o - fc5_o_fxp
    min = fc5_o_fxp_err.min().item()
    max = fc5_o_fxp_err.max().item()
    rmse = torch.sqrt(torch.mean(torch.square(fc5_o_fxp_err)))
    std, mean = torch.std_mean(fc5_o_fxp_err)
    log = "{:30s}".format("fc5 fxp") + \
            ", Absolute Error range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
            ", std," + "{:12f}".format(std) + \
            ", mean," + "{:12f}".format(mean) + \
            ", rmse," + "{:12f}".format(rmse)
    print(log)
    fp.write(log+"\n")
    err_array[bitwidth_index, 1, 3] = rmse.cpu().item()

    fc6_o_fxp_err = fc6_o - fc6_o_fxp
    min = fc6_o_fxp_err.min().item()
    max = fc6_o_fxp_err.max().item()
    rmse = torch.sqrt(torch.mean(torch.square(fc6_o_fxp_err)))
    std, mean = torch.std_mean(fc6_o_fxp_err)
    log = "{:30s}".format("fc6 fxp") + \
            ", Absolute Error range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
            ", std," + "{:12f}".format(std) + \
            ", mean," + "{:12f}".format(mean) + \
            ", rmse," + "{:12f}".format(rmse)
    print(log)
    fp.write(log+"\n")
    err_array[bitwidth_index, 1, 4] = rmse.cpu().item()

    print()

print(err_array)
fp.write(str(err_array)+"\n")
fp.close()

font = {'family':'Times New Roman', 'size': 6}
matplotlib.rc('font', **font)

my_dpi = 300
fig_h = 0.8
fig_w = 3.5
alpha = 1

labels = [str(bitwidth) for bitwidth in bitwidth_list]
x = np.arange(len(labels))  # the label locations

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=my_dpi)

# for idx in range(len(bitwidth_list)):
#     data_hub = err_array[idx, 0, :]
#     data_fxp = err_array[idx, 1, :]
#     interval = 1/2**((10).bit_length())
#     x_axe_0 = [(x[idx] - (10 - 1) / 2 * interval + x_tick * interval) for x_tick in range(5)]
#     x_axe_1 = [(x[idx] + 1 / 2 * interval + x_tick * interval) for x_tick in range(5)]
#     if idx == 0:
#         ax.bar(x_axe_0, data_hub, interval, hatch = None, label="TC", alpha=alpha, color="#7A81FF")
#         ax.bar(x_axe_1, data_fxp, interval, hatch = None, label="FXP", alpha=alpha, color="#FF7F7F")
#     else:
#         ax.bar(x_axe_0, data_hub, interval, hatch = None, alpha=alpha, color="#7A81FF")
#         ax.bar(x_axe_1, data_fxp, interval, hatch = None, alpha=alpha, color="#FF7F7F")

for idx in range(len(bitwidth_list)):
    data_hub = err_array[idx, 0, :]
    data_fxp = err_array[idx, 1, :]
    interval = 1/2**((5).bit_length())
    x_axe = [(x[idx] - (5 - 1) / 2 * interval + x_tick * interval) for x_tick in range(5)]
    if idx == 0:
        ax.plot(x_axe, data_hub, "-s", label="TC", alpha=alpha, color="#7A81FF", lw=0.5, ms=1)
        ax.plot(x_axe, data_fxp, "-^", label="FXP", alpha=alpha, color="#FF7F7F", lw=0.5, ms=1)
    else:
        ax.plot(x_axe, data_hub, "-s", alpha=alpha, color="#7A81FF", lw=0.5, ms=1)
        ax.plot(x_axe, data_fxp, "-^", alpha=alpha, color="#FF7F7F", lw=0.5, ms=1)

locs = [0, 0.01, 0.02]
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
fig.savefig(output_dir+"layer_eval_tc.pdf", bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)
