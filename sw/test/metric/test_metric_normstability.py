# %%
%load_ext autoreload
%autoreload 2

# %%
import torch
from UnarySim.sw.metric.metric import NormStability, Stability
from UnarySim.sw.stream.gen import RNG, SourceGen, BSGen
import random
import time

# %%
width = 8
length = 2**width
device = torch.device("cpu")

dim = 1
val = [1.0*_/length/length for _ in range(length**2+1)]
# val = [0.0, 0.00390625, 0.0078125]
val_tensor = torch.tensor([val])
val_bin = SourceGen(val_tensor, width, "unipolar")
rng = RNG(width, dim, "Sobol").to(device)
bs = BSGen(val_bin(), rng())
normstb = NormStability(val_tensor, mode="unipolar", threshold=0.05).to(device)
stb = Stability(val_tensor, mode="unipolar", threshold=0.05).to(device)
start_time = time.time()
for _ in range(length):
    a = bs(torch.tensor([_]))
    normstb.Monitor(a)
    stb.Monitor(a)
print("norm stability:", normstb())
print()
print("stability:", stb())
print("--- %s seconds ---" % (time.time() - start_time))

# %%
