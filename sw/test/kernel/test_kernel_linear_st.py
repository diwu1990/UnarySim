# %%
%load_ext autoreload
%autoreload 2

# %%
import torch
from UnarySim.sw.kernel.linear_st import LinearST
from UnarySim.sw.stream.gen import RNG, SourceGen, BSGen
from UnarySim.sw.metric.metric import ProgressiveError
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummaryX import summary
import matplotlib.pyplot as plt
import time
import os

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
def linear_st_test(rng="Sobol", in_feature=128, out_feature=10000, rng_width=8, bias=True, population=2, rng_stride=1):
    modes = ["bipolar", "unipolar"]
    
    for mode in modes:
        print("= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ")
        print("mode: ", mode)
        print("= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ")
        length = 2**rng_width
        ref = torch.nn.Linear(in_feature, out_feature, bias=bias).to(device)

        if mode == "unipolar":
            ref.weight.data = torch.randn(out_feature, in_feature).mul(length).round().div(length).clamp(0, 1).to(device)
            if bias is True:
                ref.bias.data = torch.randn(1, out_feature).mul(length).round().div(length).clamp(0, 1).to(device)
            iVec = ((torch.randn(1, in_feature)*length).round()/length).clamp(0, 1).to(device)
        elif mode == "bipolar":
            ref.weight.data = torch.randn(out_feature, in_feature).mul(length).round().div(length).clamp(-1, 1).to(device)
            if bias is True:
                ref.bias.data = torch.randn(1, out_feature).mul(length).round().div(length).clamp(-1, 1).to(device)
            iVec = ((torch.randn(1, in_feature)*length).round()/length).clamp(-1, 1).to(device)

        dut = LinearST(in_feature, out_feature, ref.weight, ref.bias, bias=bias, 
                       mode=mode, rng=rng, rng_width=rng_width, rng_stride=rng_stride, population=population).to(device)
        
        with torch.no_grad():
            oVec = ref(iVec)

            oVec_dut = dut(iVec)

#             print("ref out: " ,oVec)
#             print("dut out: ", oVec_dut)
            num_error = (oVec - oVec_dut)
            error = (oVec - oVec_dut) / in_feature
#             print("error: ", error)
            print("error min, max:", error.min(), error.max())
            print("num_error min, max:", num_error.min(), num_error.max())
            
            plt.plot(iVec.cpu()[0].numpy())
            plt.show()
#             to_plot = oVec.cpu().detach().numpy().flatten()
#             print("oVec distribution=========>")
#             plt.figure(figsize=(3,1.5))
#             fig = plt.hist(to_plot, bins='auto')  # arguments are passed to np.histogram
#             plt.title("data: "+mode)
#             plt.show()
            
#             to_plot = error.cpu().detach().numpy().flatten()
#             print("error distribution=========>")
#             plt.figure(figsize=(3,1.5))
#             fig = plt.hist(to_plot, bins='auto')  # arguments are passed to np.histogram
#             plt.title("data: "+mode)
#             plt.show()
            print()

# %%
rng = "Sobol"
in_feature = 512
out_feature = 1000
rng_width = 16
bias = True
population = 1
rng_stride = 3
linear_st_test(rng=rng, in_feature=in_feature, out_feature=out_feature, rng_width=rng_width, bias=bias, population=population, rng_stride=rng_stride)

# %%
