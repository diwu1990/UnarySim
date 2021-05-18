# %%
import torch
from UnarySim.sw.kernel.linear import *
import matplotlib.pyplot as plt
import time
import math
import numpy as np

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
def test(rounding = "round", abs_err = True):
    ufc_err_min_list = []
    ufc_err_max_list = []
    ufc_err_mean_list = []
    ufc_err_std_list = []
    
    ofc_err_min_list = []
    ofc_err_max_list = []
    ofc_err_mean_list = []
    ofc_err_std_list = []
    
    ifc_err_min_list = []
    ifc_err_max_list = []
    ifc_err_mean_list = []
    ifc_err_std_list = []
    
    x_label = []
    
    for bitwidth in range(6, 13):
        cycle = 2**(bitwidth-1)
        
        in_feature = 2
        out_feature = 2**12
        bias = False
        
        input = torch.cat(2*[(torch.arange(0, out_feature)/out_feature - 0.5).unsqueeze(1)], 1).to(device)
        input[:, 1] = 0.

        fc = torch.nn.Linear(in_feature, out_feature, bias=bias).to(device)
        fc.weight.data = torch.cat(2*[(torch.arange(0, out_feature)/out_feature - 0.5).unsqueeze(1)], 1).to(device)
        fc.weight.data[:, 1] = 0.
        fc_o = fc(input)

        ufc = HUBLinear(in_feature, out_feature, bias=bias, binary_weight=fc.weight.data, binary_bias=fc.bias, cycle=cycle, rounding=rounding).to(device)
        ufc_o = ufc(input)
        
        ofc = FxpLinear(in_feature, out_feature, bias=bias, binary_weight=fc.weight.data, binary_bias=fc.bias, bitwidth=bitwidth, keep_res="output", more_res="input", rounding=rounding).to(device)
        ofc_o = ofc(input)
        
        ifc = FxpLinear(in_feature, out_feature, bias=bias, binary_weight=fc.weight.data, binary_bias=fc.bias, bitwidth=bitwidth, keep_res="input",  more_res="input", rounding=rounding).to(device)
        ifc_o = ifc(input)
        
        if abs_err is True:
            ufc_err = (ufc_o - fc_o)
            ofc_err = (ofc_o - fc_o)
            ifc_err = (ifc_o - fc_o)
        else:
            ufc_err = (ufc_o - fc_o) / fc_o
            ofc_err = (ofc_o - fc_o) / fc_o
            ifc_err = (ifc_o - fc_o) / fc_o
        
        ufc_err_min_list.append(np.nanmin(ufc_err.cpu().detach().numpy()))
        ufc_err_max_list.append(np.nanmax(ufc_err.cpu().detach().numpy()))
        ufc_err_mean_list.append(np.nanmean(np.abs(ufc_err.cpu().detach().numpy())))
        ufc_err_std_list.append(np.nanstd(ufc_err.cpu().detach().numpy()))
        
        ofc_err_min_list.append(np.nanmin(ofc_err.cpu().detach().numpy()))
        ofc_err_max_list.append(np.nanmax(ofc_err.cpu().detach().numpy()))
        ofc_err_mean_list.append(np.nanmean(np.abs(ofc_err.cpu().detach().numpy())))
        ofc_err_std_list.append(np.nanstd(ofc_err.cpu().detach().numpy()))
        
        ifc_err_min_list.append(np.nanmin(ifc_err.cpu().detach().numpy()))
        ifc_err_max_list.append(np.nanmax(ifc_err.cpu().detach().numpy()))
        ifc_err_mean_list.append(np.nanmean(np.abs(ifc_err.cpu().detach().numpy())))
        ifc_err_std_list.append(np.nanstd(ifc_err.cpu().detach().numpy()))
        
        x_label.append(2**(bitwidth-1))
    return ufc_err_min_list, ufc_err_max_list, ufc_err_mean_list, ufc_err_std_list, ofc_err_min_list, ofc_err_max_list, ofc_err_mean_list, ofc_err_std_list, ifc_err_min_list, ifc_err_max_list, ifc_err_mean_list, ifc_err_std_list, x_label


# %%
rounding = "round"
abs_err = True
ufc_err_min_list, ufc_err_max_list, ufc_err_mean_list, ufc_err_std_list, ofc_err_min_list, ofc_err_max_list, ofc_err_mean_list, ofc_err_std_list, ifc_err_min_list, ifc_err_max_list, ifc_err_mean_list, ifc_err_std_list, x_label = test(rounding, abs_err)
print(ufc_err_mean_list)
print(ufc_err_std_list)
print()

print(ofc_err_mean_list)
print(ofc_err_std_list)
print()

print(ifc_err_mean_list)
print(ifc_err_std_list)
print()

print(x_label)

# %%
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

font = {'family':'Times New Roman', 'size': 6}

matplotlib.rc('font', **font)

my_dpi = 300
fig_h = 1
fig_w = 3.3115

# construct some data like what you have:
x = np.array([i for i in range(len(ufc_err_mean_list))])
means1 = np.array(ufc_err_mean_list)
stds1 = np.array(ufc_err_std_list)
mins1 = np.array(ufc_err_min_list)
maxs1 = np.array(ufc_err_max_list)

means2 = np.array(ofc_err_mean_list)
stds2 = np.array(ofc_err_std_list)
mins2 = np.array(ofc_err_min_list)
maxs2 = np.array(ofc_err_max_list)

means3 = np.array(ifc_err_mean_list)
stds3 = np.array(ifc_err_std_list)
mins3 = np.array(ifc_err_min_list)
maxs3 = np.array(ifc_err_max_list)

x_label = ['6-32', '7-64', '8-128', '9-256', '10-512', '11-1024', '12-2048']

width = 0.20
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

ax.plot(x, means1, "-o", label="uSystolic", color="#7A81FF", ms=4)
ax.fill_between(x, means1-stds1, means1+stds1, alpha=0.3, color="#7A81FF", edgecolor=None)

ax.plot(x, means2, "-s", label="FXP-o-res", color="#FF7F7F", ms=4)
ax.fill_between(x, means2-stds2, means2+stds2, alpha=0.3, color="#FF7F7F", edgecolor=None)

ax.plot(x, means3, "-^", label="FXP-i-res", color="#D783FF", ms=4)
ax.fill_between(x, means3-stds3, means3+stds3, alpha=0.3, color="#D783FF", edgecolor=None)

ax.set_xticks(x)
ax.set_xticklabels(x_label)
ax.set_yscale('linear')
ax.set_yticks([0, 0.01, 0.02])
ax.set_yticklabels(["0.00", "0.01", "0.02"])
ax.legend(loc="upper right", ncol=3, frameon=False)

fig.tight_layout()
plt.show()
fig.savefig("test_kernel_linear_fxp_hub_compare_abs_err.pdf", bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)


# %%
rounding = "round"
abs_err = False
ufc_err_min_list, ufc_err_max_list, ufc_err_mean_list, ufc_err_std_list, ofc_err_min_list, ofc_err_max_list, ofc_err_mean_list, ofc_err_std_list, ifc_err_min_list, ifc_err_max_list, ifc_err_mean_list, ifc_err_std_list, x_label = test(rounding, abs_err)
print(ufc_err_mean_list)
print(ufc_err_std_list)
print()

print(ofc_err_mean_list)
print(ofc_err_std_list)
print()

print(ifc_err_mean_list)
print(ifc_err_std_list)
print()

print(x_label)

# %%
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

font = {'family':'Times New Roman', 'size': 6}

matplotlib.rc('font', **font)

my_dpi = 300
fig_h = 1
fig_w = 3.3115

# construct some data like what you have:
x = np.array([i for i in range(len(ufc_err_mean_list))])
means1 = np.array(ufc_err_mean_list)
stds1 = np.array(ufc_err_std_list)
mins1 = np.array(ufc_err_min_list)
maxs1 = np.array(ufc_err_max_list)

means2 = np.array(ofc_err_mean_list)
stds2 = np.array(ofc_err_std_list)
mins2 = np.array(ofc_err_min_list)
maxs2 = np.array(ofc_err_max_list)

means3 = np.array(ifc_err_mean_list)
stds3 = np.array(ifc_err_std_list)
mins3 = np.array(ifc_err_min_list)
maxs3 = np.array(ifc_err_max_list)

x_label = ['6-32', '7-64', '8-128', '9-256', '10-512', '11-1024', '12-2048']

width = 0.20
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

ax.plot(x, means1, "-o", label="uSystolic", color="#7A81FF", ms=4)
ax.fill_between(x, means1-stds1, means1+stds1, alpha=0.3, color="#7A81FF", edgecolor=None)

ax.plot(x, means2, "-s", label="FXP-o-res", color="#FF7F7F", ms=4)
ax.fill_between(x, means2-stds2, means2+stds2, alpha=0.3, color="#FF7F7F", edgecolor=None)

ax.plot(x, means3, "-^", label="FXP-i-res", color="#D783FF", ms=4)
ax.fill_between(x, means3-stds3, means3+stds3, alpha=0.3, color="#D783FF", edgecolor=None)

ax.set_xticks(x)
ax.set_xticklabels(x_label)
ax.set_yscale('linear')
ax.set_yticks([0, 0.4, 0.8])
ax.set_yticklabels(["0.00", "0.40", "0.80"])
# ax.legend(loc="upper right", ncol=3, frameon=False)

fig.tight_layout()
plt.show()
fig.savefig("test_kernel_linear_fxp_hub_compare_rel_err.pdf", bbox_inches='tight', dpi=my_dpi, pad_inches=0.02)


# %%
