#! /usr/bin/python3

from binary_model import *
import torch
import torch.nn as nn
import torch.nn.functional as F


input = torch.randn(20, 3, 10).cuda()
hx1 = torch.randn(3, 20).cuda()
hx2 = hx1.clone().detach().cuda()
hx3 = hx1.clone().detach().cuda()
output1 = []
output2 = []
output3 = []

rnn1 = nn.GRUCell(10, 20).cuda()
for i in range(20):
    hx1 = rnn1(input[i], hx1)
    output1.append(hx1)

rnn2 = HardGRUCell(10, 20, hard=False).cuda()
rnn2.weight_ih.data = rnn1.weight_ih
rnn2.weight_hh.data = rnn1.weight_hh
rnn2.bias_ih.data = rnn1.bias_ih
rnn2.bias_hh.data = rnn1.bias_hh
for i in range(20):
    hx2 = rnn2(input[i], hx2)
    output2.append(hx2)


rnn3 = nn.GRU(10, 20).cuda()
rnn3.weight_ih_l0.data = rnn1.weight_ih
rnn3.weight_hh_l0.data = rnn1.weight_hh
rnn3.bias_ih_l0.data = rnn1.bias_ih
rnn3.bias_hh_l0.data = rnn1.bias_hh
output3, _ = rnn3(input, hx3.unsqueeze(dim=0))


print("Output difference between nn.GRUCell and HardGRUCell:")
for i in range(20):
    print(torch.max(torch.abs(output1[i] - output2[i])))
print()

print("Output difference between nn.GRUCell and nn.GRU:")
for i in range(20):
    print(torch.max(torch.abs(output1[i] - output3[i])))
    