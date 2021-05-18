#! /usr/bin/python3

from binary_model import *
import torch
import torch.nn as nn
import torch.nn.functional as F


rnn = nn.GRUCell(10, 20).cuda()
input = torch.randn(20, 3, 10).cuda()
hx = torch.randn(3, 20).cuda()
hx2 = hx.clone().detach().cuda()
output = []
for i in range(20):
    hx = rnn(input[i], hx)
    output.append(hx)

custrnn = HardGRUCell(10, 20, hard=False).cuda()
custrnn.weight_ih.data = rnn.weight_ih
custrnn.weight_hh.data = rnn.weight_hh
custrnn.bias_ih.data = rnn.bias_ih
custrnn.bias_hh.data = rnn.bias_hh
output2 = []
for i in range(20):
    hx2 = custrnn(input[i], hx2)
    output2.append(hx2)

for i in range(20):
    print(torch.max(torch.abs(output[i] - output2[i])))