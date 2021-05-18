# %%
%load_ext autoreload
%autoreload 2

# %%
import torch
from UnarySim.sw.metric.metric import Correlation

# %%
corr = Correlation().cuda()

# %%
a = torch.tensor([0]).type(torch.int8).cuda()
b = torch.tensor([0]).type(torch.int8).cuda()

# %%
a = torch.tensor([0]).type(torch.int8).cuda()
b = torch.tensor([1]).type(torch.int8).cuda()

# %%
a = torch.tensor([1]).type(torch.int8).cuda()
b = torch.tensor([0]).type(torch.int8).cuda()

# %%
a = torch.tensor([1]).type(torch.int8).cuda()
b = torch.tensor([1]).type(torch.int8).cuda()

# %%
corr.Monitor(a,b)
print("d", corr.paired_00_d)
print("c", corr.paired_01_c)
print("b", corr.paired_10_b)
print("a", corr.paired_11_a)

cor = corr()
print(cor)

# %%
a = torch.tensor([0]).type(torch.int8).cuda()

# %%
a = torch.tensor([1]).type(torch.int8).cuda()

# %%
corr.Monitor(a)
print("d", corr.paired_00_d)
print("c", corr.paired_01_c)
print("b", corr.paired_10_b)
print("a", corr.paired_11_a)

cor = corr()
print("correlation", cor)

# %%
