import torch
from torch import nn as nn

input = torch.ones(1, 2, 10, 10)
conv = nn.Conv2d(2, 1, 3, dilation=2, bias=False)
conv.weight.data.fill_(1.)
output = conv(input)

print(input.shape)
print(input)
print(output.shape)
print(output)
