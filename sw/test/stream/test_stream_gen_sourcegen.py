# %%
%load_ext autoreload
%autoreload 2

# %%
import torch
from UnarySim.sw.stream.gen import SourceGen

# %%
srcbin = SourceGen(torch.tensor([0.5, 0.7]).cpu(), 8, "unipolar").cuda()
print(srcbin)
print(srcbin())

# %%
srcbin2 = SourceGen(torch.tensor([0.5, 0.7]), 8, "bipolar")
print(srcbin2())

# %%
