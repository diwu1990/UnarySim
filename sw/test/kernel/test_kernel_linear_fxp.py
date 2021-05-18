# %%
import torch
from UnarySim.sw.kernel.linear import *
import matplotlib.pyplot as plt
import time
import math

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
in_feature = 16
out_feature = 8
bias = True
bitwidth = 8
keep_res = "input"
more_res = "input"
rounding = "round"

total_bit = 16
input_int_bit = 3
input_fra_bit = total_bit - input_int_bit

input = ((torch.rand(256, in_feature) - 0.5) * 2**(2*input_int_bit)).round().div(2**(input_int_bit)).to(device)

fc = torch.nn.Linear(in_feature, out_feature, bias=bias).to(device)
fc_o = fc(input)

ufc = FxpLinear(in_feature, out_feature, bias=bias, binary_weight=fc.weight.data, binary_bias=fc.bias, bitwidth=bitwidth, keep_res=keep_res, more_res=more_res, rounding=rounding).to(device)
ufc_o = ufc(input)

(fc_o - ufc_o).abs().mean().backward()

# %%
diff = (ufc_o - fc_o)
print("diff max:", diff.max())
print("diff min:", diff.min())
print("diff mean:", diff.mean())

fig = plt.hist(diff.cpu().detach().numpy().flatten(), bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram for output error")
plt.show()

# %%
diff_grad = (ufc.weight.grad - fc.weight.grad)
print("diff grad max:", diff_grad.max())
print("diff grad min:", diff_grad.min())
print("diff grad mean:", diff_grad.mean())

fig = plt.hist(diff_grad.cpu().detach().numpy().flatten(), bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram for grad error")
plt.show()

# %%
print(ufc_o)

# %%
