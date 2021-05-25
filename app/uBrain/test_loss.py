import torch
from torch import nn as nn
from scipy.stats import truncnorm
import matplotlib.pyplot as plt


m = nn.LogSoftmax(dim=1)

criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.NLLLoss()

x = torch.randn(1, 5)
y = torch.empty(1, dtype=torch.long).random_(5)

loss1 = criterion1(x, y)
loss2 = criterion2(m(x), y)
print(loss1)
print(loss2)

sm = torch.nn.Softmax()(x)
log = torch.log(sm)
sum = -torch.sum(y * log, dim=1)
loss3 = torch.mean(sum)
print(loss3)
