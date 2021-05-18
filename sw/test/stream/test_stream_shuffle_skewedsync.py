# %%
%load_ext autoreload
%autoreload 2

# %%
import torch
import UnarySim
from UnarySim.sw.stream.shuffle import SkewedSync


# %%
ss = SkewedSync(depth=3).cuda()

# %%
print(ss.cnt)

# %%
a = torch.tensor([[0, 0]]).type(torch.float)
b = torch.tensor([[1, 1]]).type(torch.float)

# %%
a = torch.tensor([[1, 1]]).type(torch.float)
b = torch.tensor([[0, 0]]).type(torch.float)

# %%
a = torch.tensor([[1, 1]]).type(torch.float)
b = torch.tensor([[1, 1]]).type(torch.float)

# %%
a = torch.tensor([[0, 0]]).type(torch.float)
b = torch.tensor([[0, 0]]).type(torch.float)

# %%
a = a.cuda()
print(a)
b = b.cuda()
print(b)


# %%
print(ss(a,b))
print(ss.cnt)

# %%


# %%


# %%


# %%
