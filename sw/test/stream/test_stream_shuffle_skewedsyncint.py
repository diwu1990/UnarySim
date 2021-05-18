# %%
import torch
import UnarySim
from UnarySim.sw.stream.shuffle_int import SkewedSyncInt

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
ss = SkewedSyncInt(depth=3).to(device)

# %%
a = torch.tensor([[0, 0]]).type(torch.float).to(device)
b = torch.tensor([[1, 1]]).type(torch.float).to(device)
out_a, out_b = ss(a,b)
print(out_a)
print(out_b)
print(ss.cnt)

# %%
a = torch.tensor([[4, 4]]).type(torch.float).to(device)
b = torch.tensor([[0, 0]]).type(torch.float).to(device)
out_a, out_b = ss(a,b)
print(out_a)
print(out_b)
print(ss.cnt)

# %%
a = torch.tensor([[4, 4]]).type(torch.float).to(device)
b = torch.tensor([[1, 1]]).type(torch.float).to(device)
out_a, out_b = ss(a,b)
print(out_a)
print(out_b)
print(ss.cnt)

# %%
a = torch.tensor([[0, 0]]).type(torch.float).to(device)
b = torch.tensor([[0, 0]]).type(torch.float).to(device)
out_a, out_b = ss(a,b)
print(out_a)
print(out_b)
print(ss.cnt)

# %%
a = torch.tensor([[4, 4]]).type(torch.float).to(device)
b = torch.tensor([[1, 1]]).type(torch.float).to(device)
out_a, out_b = ss(a,b)
print(out_a)
print(out_b)
print(ss.cnt)

# %%
a = torch.tensor([[4, 4]]).type(torch.float).to(device)
b = torch.tensor([[0, 0]]).type(torch.float).to(device)
out_a, out_b = ss(a,b)
print(out_a)
print(out_b)
print(ss.cnt)

# %%
a = torch.tensor([[4, 4]]).type(torch.float).to(device)
b = torch.tensor([[0, 0]]).type(torch.float).to(device)
out_a, out_b = ss(a,b)
print(out_a)
print(out_b)
print(ss.cnt)

# %%
a = torch.tensor([[4, 4]]).type(torch.float).to(device)
b = torch.tensor([[0, 0]]).type(torch.float).to(device)
out_a, out_b = ss(a,b)
print(out_a)
print(out_b)
print(ss.cnt)

# %%
a = torch.tensor([[2, 2]]).type(torch.float).to(device)
b = torch.tensor([[0, 0]]).type(torch.float).to(device)
out_a, out_b = ss(a,b)
print(out_a)
print(out_b)
print(ss.cnt)