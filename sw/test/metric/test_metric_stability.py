# %%
import torch
from UnarySim.sw.metric.metric import Stability


# %%
input = torch.tensor([-0.5,0]).cuda()

# %%
stb = Stability(input, mode="bipolar", threshold=0.1).cuda()

# %%
a = torch.tensor([1,0]).type(torch.int8).cuda()

# %%
a = torch.tensor([0,1]).type(torch.int8).cuda()

# %%
stb.Monitor(a)
print(stb.err)
print(stb.stable_len)
print(stb.len)
print(stb.threshold)
print(stb())

# %%
