import torch
from torch import nn as nn

kernel_size = (3, 3)
dilation = 1
padding = 1
stride = 1

input = torch.ones(2, 1, 5, 5)
padding0 = torch.nn.ConstantPad2d(padding, 0)
input0 = padding0(input)
output0 = torch.nn.functional.unfold(input0, kernel_size, dilation, 0, stride)

output1 = torch.nn.functional.unfold(input, kernel_size, dilation, padding, stride)

print(output0.shape)
print(output0)

print(output1.shape)
print(output1)

print(torch.sum(output0 == output1) == torch.prod(torch.tensor(output1.size())))