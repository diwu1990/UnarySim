import math
import torch
from UnarySim.kernel import FSUConv2d
from UnarySim.stream import RNG, BinGen, BSGen
from UnarySim.metric import ProgError
import matplotlib.pyplot as plt
import time
import torch.autograd.profiler as profiler
from UnarySim.kernel import conv2d_output_shape, num2tuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_fsuconv2d():
    plot_en=False

    hwcfg_input={
        "width" : 8,
        "rng" : "Sobol",
        "dimr" : 1
    }
    hwcfg={
        "width" : 8,
        "mode" : "bipolar",
        "scale" : None,
        "depth" : 20,
        "rng" : "Sobol",
        "dimr" : 1
    }
    swcfg={
        "btype" : torch.float, 
        "rtype" : torch.float, 
        "stype" : torch.float
    }

    rng = hwcfg["rng"]
    
    in_channels=32 
    out_channels=16 
    kernel_size=3 
    stride=2 
    padding=0 
    dilation=1 
    groups=1 
    bias=True 
    padding_mode='zeros' 

    modes = ["bipolar", "unipolar"]
    scaled = [True, False]
    result_pe = []
    
    for mode in modes:
        for scale in scaled:
            hwcfg["mode"] = mode
            hwcfg_input["mode"] = mode
            hwcfg["scale"] = (kernel_size * kernel_size * in_channels + bias) if scale else 1

            length = 2**hwcfg["width"]
            length_input = 2**hwcfg_input["width"]
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
                                weight_ext=conv2d.weight, bias_ext=conv2d.bias, hwcfg=hwcfg, swcfg=swcfg).to(device)

            input_size = (128, 32)
            iVec = ((torch.rand(32, in_channels, input_size[0], input_size[1])*length_input).round()/length_input).to(device)
            oVec = conv2d(iVec)

            iVecSource = BinGen(iVec, hwcfg, swcfg)().to(device)
            iVecRNG = RNG(hwcfg_input, swcfg)().to(device)
            iVecBS = BSGen(iVecSource, iVecRNG, swcfg).to(device)

            hwcfg["scale"] = 1
            iVecPE = ProgError(iVec, hwcfg).to(device)

            hwcfg["scale"] = (kernel_size * kernel_size * in_channels + bias) if scale else 1
            oVecPE = ProgError(oVec, hwcfg).to(device)
            
            with torch.no_grad():
                idx = torch.zeros(iVecSource.size()).type(torch.long).to(device)
                start_time = time.time()
                for i in range(length):
                    iBS = iVecBS(idx + i)
                    iVecPE.Monitor(iBS)

                    oVecU = uconv2d(iBS)
                    oVecPE.Monitor(oVecU)
                    rmse = torch.sqrt(torch.sum(torch.mul(oVecPE()[1], oVecPE()[1]))/torch.prod(torch.tensor(oVecPE()[1].size())))
                    if plot_en is True:
                        result_pe_cycle.append(1-rmse.item())
                print("--- %s seconds ---" % (time.time() - start_time))
                print("RNG: "+rng+", data: "+mode+", scaled: "+str(scale))
                print("input error:  ", "min: ", torch.min(iVecPE()[1]).item(), "max: ", torch.max(iVecPE()[1]).item())
                print("output error: ", "min: ", torch.min(oVecPE()[1]).item(), "max: ", torch.max(oVecPE()[1]).item(), "RMSE: ", rmse.item())
                print()
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


if __name__ == '__main__':
    test_fsuconv2d()

