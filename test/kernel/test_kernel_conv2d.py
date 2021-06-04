# %%
import math
import torch
from UnarySim.kernel.conv import FSUConv2d
from UnarySim.stream.gen import RNG, SourceGen, BSGen
from UnarySim.metric.metric import ProgError
import matplotlib.pyplot as plt
import time
import torch.autograd.profiler as profiler

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
def linear_test(rng="Sobol", 
                in_channels=32, 
                out_channels=16, 
                kernel_size=3, 
                stride=2, 
                padding=0, 
                dilation=1, 
                groups=1, 
                bias=True, 
                padding_mode='zeros', 
                bitwidth=8,
                plot_en=False):
    modes = ["bipolar", "unipolar"]
    scaled = [True, False]
    result_pe = []
    
    for mode in modes:
        for scale in scaled:
            length = 2**bitwidth
            result_pe_cycle = []
            conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode).to(device)
            
            if mode == "unipolar":
                conv2d.weight.data = torch.rand(out_channels, in_channels, kernel_size, kernel_size).mul(length).round().div(length).to(device)
                if bias is True:
                    conv2d.bias.data = torch.rand(out_channels).mul(length).round().div(length).to(device)
            elif mode == "bipolar":
                conv2d.weight.data = torch.rand(out_channels, in_channels, kernel_size, kernel_size).mul(2).sub(1).mul(length).round().div(length).to(device)
                if bias is True:
                    conv2d.bias.data = torch.rand(out_channels).mul(2).sub(1).mul(length).round().div(length).to(device)

            uconv2d = FSUConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, 
                                binary_weight=conv2d.weight, binary_bias=conv2d.bias, bitwidth=bitwidth, mode=mode, scaled=scale).to(device)

            input_size = (128, 32)
            iVec = ((torch.rand(32, in_channels, input_size[0], input_size[1])*length).round()/length).to(device)
            oVec = conv2d(iVec)

            iVecSource = SourceGen(iVec, bitwidth=bitwidth, mode=mode)().to(device)
            iVecRNG = RNG(bitwidth, 1, rng)().to(device)
            iVecBS = BSGen(iVecSource, iVecRNG).to(device)

            iVecPE = ProgError(iVec, mode=mode).to(device)
            
            if scale is True:
                if bias == 0:
                    oVecPE = ProgError(oVec, scale=kernel_size * kernel_size * in_channels, mode=mode).to(device)
                elif bias ==1:
                    oVecPE = ProgError(oVec, scale=kernel_size * kernel_size * in_channels+1, mode=mode).to(device)
            else:
                oVecPE = ProgError(oVec, scale=1, mode=mode).to(device)
            
            with torch.no_grad():
                idx = torch.zeros(iVecSource.size()).type(torch.long).to(device)
                start_time = time.time()
                for i in range(length):
                    iBS = iVecBS(idx + i)
                    iVecPE.Monitor(iBS)

                    oVecU = uconv2d(iBS)
                    oVecPE.Monitor(oVecU)
                    rmse = torch.sqrt(torch.sum(torch.mul(oVecPE()[1], oVecPE()[1]))/torch.prod(torch.tensor(oVecPE()[1].size())))
                    result_pe_cycle.append(1-rmse.item())
                print("--- %s seconds ---" % (time.time() - start_time))
                print("RNG: "+rng+", data: "+mode+", scaled: "+str(scale))
                print("input error:  ", "min: ", torch.min(iVecPE()[1]).item(), "max: ", torch.max(iVecPE()[1]).item())
                print("output error: ", "min: ", torch.min(oVecPE()[1]).item(), "max: ", torch.max(oVecPE()[1]).item(), "RMSE: ", rmse.item())
                if plot_en is True:
                    result_pe = oVecPE()[1].cpu().numpy()
                    print("error distribution=========>")
                    plt.figure(figsize=(3,1.5))
                    fig = plt.hist(result_pe.flatten(), bins='auto')  # arguments are passed to np.histogram
                    plt.show()
                    print("progressive accuracy=========>")
                    plt.figure(figsize=(3,1.5))
                    fig = plt.plot(result_pe_cycle)  # arguments are passed to np.histogram
                    plt.show()

# %%
rng = "Sobol"
in_channels=16
out_channels=128
kernel_size=3
stride=3
padding=3
dilation=4
groups=1
bias=True
padding_mode='zeros'
bitwidth=10
linear_test(rng, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, bitwidth=bitwidth)
