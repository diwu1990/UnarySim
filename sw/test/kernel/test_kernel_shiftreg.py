# %%
import torch
from UnarySim.sw.kernel.shiftreg import ShiftReg
from UnarySim.sw.stream.gen import RNG, SourceGen, BSGen
from UnarySim.sw.metric.metric import ProgressiveError
import matplotlib.pyplot as plt
import time
import math
import numpy as np

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
depth = 4
stype = torch.int8
sr = ShiftReg(depth=4, stype=stype).to(device)

# %%
a = torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0]]).type(stype).to(device)

# %%
oBit, cnt = sr(a, mask=torch.tensor([[1, 1], [1, 1], [0, 0], [0, 0]]).to(device))
print(oBit, cnt)
print(sr.sr)

# %%
