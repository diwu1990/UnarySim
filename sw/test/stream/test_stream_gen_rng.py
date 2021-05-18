# %%
%load_ext autoreload
%autoreload 2

# %%
import torch
import UnarySim
from UnarySim.sw.stream.gen import RNG

import logging
log = logging.getLogger(__name__)
# log.error("asdad{}".format("asda"))

# %%
rng = RNG(1, 2, "Sobol")
print(rng.cuda()())
print(rng.cpu()())

# %%
rng = RNG(4, 2, "Race")
print(rng.cuda()())
print(rng.cpu()())

# %%
rng = RNG(4, 2, "LFSR")
print(rng.cuda()())
print(rng.cpu()())

# %%
rng = RNG(4, 2, "SYS")
print(rng.cuda()())
print(rng.cpu()())

# %%
