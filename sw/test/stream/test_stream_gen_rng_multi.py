# %%
%load_ext autoreload
%autoreload 2

# %%
import torch
import UnarySim
from UnarySim.sw.stream.gen import RNGMulti

import logging
log = logging.getLogger(__name__)
# log.error("asdad{}".format("asda"))

# %%
rng = RNGMulti(4, 2, "Sobol", True)
print(rng.cuda()())
print(rng.cpu()())

# %%
rng = RNGMulti(4, 2, "LFSR", True)
print(rng.cuda()())
print(rng.cpu()())

# %%
rng = RNGMulti(4, 2, "SYS", True)
print(rng.cuda()())
print(rng.cpu()())

# %%
