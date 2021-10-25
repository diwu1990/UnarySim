#! /usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from UnarySim.kernel.rnn import FSUMGUCell, HardMGUCell, HardMGUCellFxp
from UnarySim.stream.gen import *
from UnarySim.metric.metric import SourceGen, RNG, BSGen, ProgError
from UnarySim.kernel.utils import truncated_normal, progerror_report
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
    rnn3 = HardMGUCellFxp(input_sz, hidden_sz, bias=bias, hard=True, intwidth=intwidth, fracwidth=fracwidth).to(device)
    rnn3.weight_f.data = rnn1.weight_f.clone().detach().to(device)
    rnn3.weight_n.data = rnn1.weight_n.clone().detach().to(device)

    for i in range(win_sz):
        hx1 = rnn1(input[i], hx1)
        output1.append(hx1)

        hx3 = rnn3(input[i], hx3)
        output3.append(hx3)

        iVec, hVec = input[i], hx2

        # rnn2 in the loop to mimic the hw reset
        rnn2 = FSUMGUCell(input_sz, hidden_sz, bias=bias, 
                        binary_weight_f=rnn1.weight_f, binary_bias_f=rnn1.bias_f, binary_weight_n=rnn1.weight_n, binary_bias_n=rnn1.bias_n, 
                        hx_buffer=hx2, 
                        bitwidth=bitwidth, mode=mode, depth=depth, depth_ismul=depth_ismul).to(device)

        iSource = SourceGen(iVec, bitwidth=bitwidth, mode=mode)().to(device)
        iRNG = RNG(bitwidth, 1, rng)().to(device)
        iBSG = BSGen(iSource, iRNG).to(device)
        iPE = ProgError(iVec, scale=1, mode=mode).to(device)

        hSource = SourceGen(hVec, bitwidth=bitwidth, mode=mode)().to(device)
        hRNG = RNG(bitwidth, 1, rng)().to(device)
        hBSG = BSGen(hSource, hRNG).to(device)
        hPE = ProgError(hVec, scale=1, mode=mode).to(device)

        oVec = output1[i]
        oPE = ProgError(oVec, scale=1, mode=mode).to(device)
        
        fg_ug_in_PE     = ProgError(rnn1.fg_ug_in,  scale=1, mode=mode).to(device)
        fg_in_PE        = ProgError(rnn1.fg_in,     scale=1, mode=mode).to(device)
        fg_PE           = ProgError(rnn1.fg,        scale=1, mode=mode).to(device)
        fg_hx_PE        = ProgError(rnn1.fg_hx,     scale=1, mode=mode).to(device)
        ng_ug_in_PE     = ProgError(rnn1.ng_ug_in,  scale=1, mode=mode).to(device)
        ng_PE           = ProgError(rnn1.ng,        scale=1, mode=mode).to(device)
        fg_ng_PE        = ProgError(rnn1.fg_ng,     scale=1, mode=mode).to(device)
        fg_ng_inv_PE    = ProgError(rnn1.fg_ng_inv, scale=1, mode=mode).to(device)

        for c in range(2**bitwidth):
            idx = torch.zeros(iSource.size()).type(torch.long).to(device)
            iBS = iBSG(idx + c)
            iPE.Monitor(iBS)

            hdx = torch.zeros(hSource.size()).type(torch.long).to(device)
            hBS = hBSG(hdx + c)
            hPE.Monitor(hBS)

            start_time = time.time()

            oBS = rnn2(iBS, hBS)

            fg_ug_in_PE.Monitor(rnn2.fg_ug_in)
            fg_in_PE.Monitor(rnn2.fg_in)
            fg_PE.Monitor(rnn2.fg)
            fg_hx_PE.Monitor(rnn2.fg_hx)
            ng_ug_in_PE.Monitor(rnn2.ng_ug_in)
            ng_PE.Monitor(rnn2.ng)
            fg_ng_PE.Monitor(rnn2.fg_ng)
            fg_ng_inv_PE.Monitor(rnn2.fg_ng_inv)

            oPE.Monitor(oBS)

        hx2 = oPE()[0]
        output2.append(hx2)

        # print("======>> window: " + str(i) + "<<======")
        # print("--- %s seconds ---" % (time.time() - start_time))
        if output_error_only:
            pass
        else:
            progerror_report(iPE, "input")
            progerror_report(hPE, "hidden")

            progerror_report(fg_ug_in_PE, "fg_ug_in")
            progerror_report(fg_in_PE, "fg_in")
            progerror_report(fg_PE, "fg")
            progerror_report(fg_hx_PE, "fg_hx")
            progerror_report(ng_ug_in_PE, "ng_ug_in")
            progerror_report(ng_PE, "ng")
            progerror_report(fg_ng_PE, "fg_ng")
            progerror_report(fg_ng_inv_PE, "fg_ng_inv")

        progerror_report(oPE, str(i)+"-th win output unary")

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
