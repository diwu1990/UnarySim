#! /usr/bin/python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from UnarySim.kernel.rnn import FSUMGUCell
from UnarySim.app.uBrain.binary_model import HardMGUCell, HardMGUCell_i, truncated_normal
from UnarySim.stream.gen import *
from UnarySim.metric.metric import *
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

win_sz = 10
batch = 32
input_sz = 256
hidden_sz = 64
bitwidth = 12
mode = "bipolar"
depth = bitwidth + 2
rng = "Sobol"
bias = False

input = torch.randn(win_sz, batch, input_sz).to(device)
input = truncated_normal(input, mean=0, std=0.4)
hx1 = torch.randn(batch, hidden_sz).to(device)
hx1 = truncated_normal(hx1, mean=0, std=0.1)
hx2 = hx1.clone().detach().to(device)
output1 = []
output2 = []

rnn1 = HardMGUCell_i(input_sz, hidden_sz, bias=bias, hard=True).to(device)

# rnn2 = FSUMGUCell(input_sz, hidden_sz, bias=bias, 
#                     binary_weight_ih=rnn1.weight_ih, binary_bias_ih=rnn1.bias_ih, binary_weight_hh=rnn1.weight_hh, binary_bias_hh=rnn1.bias_hh, 
#                     bitwidth=bitwidth, mode=mode, depth=depth).to(device)

for i in range(win_sz):
    hx1 = rnn1(input[i], hx1)
    output1.append(hx1)

    iVec, hVec = input[i], hx2

    # rnn2 in the loop to mimic the hw reset
    rnn2 = FSUMGUCell(input_sz, hidden_sz, bias=bias, 
                    binary_weight_ih=rnn1.weight_ih, binary_bias_ih=rnn1.bias_ih, binary_weight_hh=rnn1.weight_hh, binary_bias_hh=rnn1.bias_hh, 
                    bitwidth=bitwidth, mode=mode, depth=depth).to(device)

    iSource = SourceGen(iVec, bitwidth=bitwidth, mode=mode)().to(device)
    iRNG = RNG(bitwidth, 1, rng)().to(device)
    iBSG = BSGen(iSource, iRNG).to(device)
    iPE = ProgressiveError(iVec, scale=1, mode=mode).to(device)

    hSource = SourceGen(hVec, bitwidth=bitwidth, mode=mode)().to(device)
    hRNG = RNG(bitwidth, 1, rng)().to(device)
    hBSG = BSGen(hSource, hRNG).to(device)
    hPE = ProgressiveError(hVec, scale=1, mode=mode).to(device)

    oVec = output1[i]
    oPE = ProgressiveError(oVec, scale=1, mode=mode).to(device)
    
    forgetgate_in_PE    = ProgressiveError(rnn1.forgetgate_in,  scale=1, mode=mode).to(device)
    forgetgate_PE       = ProgressiveError(rnn1.forgetgate,     scale=1, mode=mode).to(device)
    h_n_hardtanh_PE     = ProgressiveError(rnn1.h_n_hardtanh,   scale=1, mode=mode).to(device)
    newgate_prod_PE     = ProgressiveError(rnn1.newgate_prod,   scale=1, mode=mode).to(device)
    i_n_hardtanh_PE     = ProgressiveError(rnn1.i_n_hardtanh,   scale=1, mode=mode).to(device)
    newgate_in_PE       = ProgressiveError(rnn1.newgate_in,     scale=1, mode=mode).to(device)
    newgate_PE          = ProgressiveError(rnn1.newgate,        scale=1, mode=mode).to(device)
    forgetgate_inv_prod_PE      = ProgressiveError(rnn1.forgetgate_inv_prod,        scale=1, mode=mode).to(device)
    forgetgate_prod_PE          = ProgressiveError(rnn1.forgetgate_prod,        scale=1, mode=mode).to(device)

    for i in range(2**bitwidth):
        idx = torch.zeros(iSource.size()).type(torch.long).to(device)
        iBS = iBSG(idx + i)
        iPE.Monitor(iBS)

        hdx = torch.zeros(hSource.size()).type(torch.long).to(device)
        hBS = hBSG(hdx + i)
        hPE.Monitor(hBS)

        start_time = time.time()

        oBS = rnn2(iBS, hBS)

        forgetgate_in_PE.Monitor(rnn2.forgetgate_in)
        forgetgate_PE.Monitor(rnn2.forgetgate)
        h_n_hardtanh_PE.Monitor(rnn2.h_n_hardtanh)
        newgate_prod_PE.Monitor(rnn2.newgate_prod)
        i_n_hardtanh_PE.Monitor(rnn2.i_n_hardtanh)
        newgate_in_PE.Monitor(rnn2.newgate_in)
        newgate_PE.Monitor(rnn2.newgate)
        forgetgate_inv_prod_PE.Monitor(rnn2.forgetgate_inv_prod)
        forgetgate_prod_PE.Monitor(rnn2.forgetgate_prod)

        oPE.Monitor(oBS)
    
    forgetgate_in_rmse = torch.sqrt(torch.mean(torch.square(forgetgate_in_PE()[1])))
    forgetgate_rmse = torch.sqrt(torch.mean(torch.square(forgetgate_PE()[1])))
    h_n_hardtanh_rmse = torch.sqrt(torch.mean(torch.square(h_n_hardtanh_PE()[1])))
    newgate_prod_rmse = torch.sqrt(torch.mean(torch.square(newgate_prod_PE()[1])))
    i_n_hardtanh_rmse = torch.sqrt(torch.mean(torch.square(i_n_hardtanh_PE()[1])))
    newgate_in_rmse = torch.sqrt(torch.mean(torch.square(newgate_in_PE()[1])))
    newgate_rmse = torch.sqrt(torch.mean(torch.square(newgate_PE()[1])))
    forgetgate_inv_prod_rmse = torch.sqrt(torch.mean(torch.square(forgetgate_inv_prod_PE()[1])))
    forgetgate_prod_rmse = torch.sqrt(torch.mean(torch.square(forgetgate_prod_PE()[1])))

    rmse = torch.sqrt(torch.mean(torch.square(oPE()[1])))

    hx2 = oPE()[0]
    output2.append(hx2)

    print("--- %s seconds ---" % (time.time() - start_time))
    print("input error:                 min: "+"{:10f}".format(torch.min(iPE()[1]).item())+"    max: "+"{:10f}".format(torch.max(iPE()[1]).item()))
    print("hidden error:                min: "+"{:10f}".format(torch.min(hPE()[1]).item())+"    max: "+"{:10f}".format(torch.max(hPE()[1]).item()))

    print("forgetgate_in error:         min: "+"{:10f}".format(torch.min(forgetgate_in_PE()[1]).item())+"    max: "+"{:10f}".format(torch.max(forgetgate_in_PE()[1]).item())+"    rsme: "+"{:10f}".format(forgetgate_in_rmse.item()))
    print("forgetgate error:            min: "+"{:10f}".format(torch.min(forgetgate_PE()[1]).item())+"    max: "+"{:10f}".format(torch.max(forgetgate_PE()[1]).item())+"    rsme: "+"{:10f}".format(forgetgate_rmse.item()))
    print("h_n_hardtanh error:          min: "+"{:10f}".format(torch.min(h_n_hardtanh_PE()[1]).item())+"    max: "+"{:10f}".format(torch.max(h_n_hardtanh_PE()[1]).item())+"    rsme: "+"{:10f}".format(h_n_hardtanh_rmse.item()))
    print("newgate_prod error:          min: "+"{:10f}".format(torch.min(newgate_prod_PE()[1]).item())+"    max: "+"{:10f}".format(torch.max(newgate_prod_PE()[1]).item())+"    rsme: "+"{:10f}".format(newgate_prod_rmse.item()))
    print("i_n_hardtanh error:          min: "+"{:10f}".format(torch.min(i_n_hardtanh_PE()[1]).item())+"    max: "+"{:10f}".format(torch.max(i_n_hardtanh_PE()[1]).item())+"    rsme: "+"{:10f}".format(i_n_hardtanh_rmse.item()))
    print("newgate_in error:            min: "+"{:10f}".format(torch.min(newgate_in_PE()[1]).item())+"    max: "+"{:10f}".format(torch.max(newgate_in_PE()[1]).item())+"    rsme: "+"{:10f}".format(newgate_in_rmse.item()))
    print("newgate error:               min: "+"{:10f}".format(torch.min(newgate_PE()[1]).item())+"    max: "+"{:10f}".format(torch.max(newgate_PE()[1]).item())+"    rsme: "+"{:10f}".format(newgate_rmse.item()))
    print("forgetgate_inv_prod error:   min: "+"{:10f}".format(torch.min(forgetgate_inv_prod_PE()[1]).item())+"    max: "+"{:10f}".format(torch.max(forgetgate_inv_prod_PE()[1]).item())+"    rsme: "+"{:10f}".format(forgetgate_inv_prod_rmse.item()))
    print("forgetgate_prod error:       min: "+"{:10f}".format(torch.min(forgetgate_prod_PE()[1]).item())+"    max: "+"{:10f}".format(torch.max(forgetgate_prod_PE()[1]).item())+"    rsme: "+"{:10f}".format(forgetgate_prod_rmse.item()))

    print("output error:                min: "+"{:10f}".format(torch.min(oPE()[1]).item())+"    max: "+"{:10f}".format(torch.max(oPE()[1]).item())+"    rsme: "+"{:10f}".format(rmse.item()))


print()
