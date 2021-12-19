#! /usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from UnarySim.kernel import HUBMGUCell, HardMGUCell, HardMGUCellFXP
from UnarySim.kernel import truncated_normal, progerror_report
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_hubmgu():
    bitwidth_list = [7, 8, 9, 10]
    for bitwidth in bitwidth_list:
        print("bit width:", bitwidth)
        win_sz = 10
        batch = 32
        input_sz = 256
        hidden_sz = 64

        intwidth = 1

        fracwidth = bitwidth - intwidth
        mode = "bipolar"
        depth = bitwidth + 2
        depth_ismul = bitwidth - 4
        rng = "Sobol"
        bias = False
        output_error_only=True

        hwcfg={
            "width" : bitwidth,
            "mode" : mode,
            "depth" : depth,
            "depth_ismul" : depth_ismul,
            "rng" : rng,
            "dimr" : 1
        }

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
                        weight_ext_f=rnn1.weight_f, bias_ext_f=rnn1.bias_f, weight_ext_n=rnn1.weight_n, bias_ext_n=rnn1.bias_n, 
                        hwcfg=hwcfg).to(device)

        rnn3 = HardMGUCellFXP(input_sz, hidden_sz, bias=bias, hard=True, intwidth=intwidth, fracwidth=fracwidth).to(device)
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
                print("{:30s}".format(str(i)+"-th win output hub") + \
                        ", Absolute Error range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
                        ", std," + "{:12f}".format(std) + \
                        ", mean," + "{:12f}".format(mean) + \
                        ", rmse," + "{:12f}".format(rmse))

                fxp_err = hx1 - hx3
                min = fxp_err.min().item()
                max = fxp_err.max().item()
                rmse = torch.sqrt(torch.mean(torch.square(fxp_err)))
                std, mean = torch.std_mean(fxp_err)
                print("{:30s}".format(str(i)+"-th win output fxp") + \
                        ", Absolute Error range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
                        ", std," + "{:12f}".format(std) + \
                        ", mean," + "{:12f}".format(mean) + \
                        ", rmse," + "{:12f}".format(rmse))

        print()


if __name__ == '__main__':
    test_hubmgu()

