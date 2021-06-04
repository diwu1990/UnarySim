# %%
import torch
from UnarySim.kernel.linear import FSULinearuGEMM
from UnarySim.stream.gen import RNG, SourceGen, BSGen
from UnarySim.metric.metric import ProgError
import matplotlib.pyplot as plt
import time
import torch.autograd.profiler as profiler

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
def linear_test(rng="Sobol", in_feature=128, out_feature=10000, bitwidth=8, bias=True, plot_en=False):
    modes = ["bipolar", "unipolar"]
    scaled = [True, False]
    result_pe = []
    
    for mode in modes:
        for scale in scaled:
            length = 2**bitwidth
            result_pe_cycle = []
            fc = torch.nn.Linear(in_feature, out_feature, bias=bias).to(device)
            
            if mode == "unipolar":
                fc.weight.data = torch.rand(out_feature, in_feature).mul(length).round().div(length).to(device)
                if bias is True:
                    fc.bias.data = torch.rand(out_feature).mul(length).round().div(length).to(device)
            elif mode == "bipolar":
                fc.weight.data = torch.rand(out_feature, in_feature).mul(2).sub(1).mul(length).round().div(length).to(device)
                if bias is True:
                    fc.bias.data = torch.rand(out_feature).mul(2).sub(1).mul(length).round().div(length).to(device)

            ufc = FSULinearuGEMM(in_feature, out_feature, bias=bias, binary_weight=fc.weight, binary_bias=fc.bias, 
                              bitwidth=bitwidth, mode=mode, scaled=scale).to(device)

            iVec = ((torch.rand(32, in_feature)*length).round()/length).to(device)
            oVec = fc(iVec)

            iVecSource = SourceGen(iVec, bitwidth=bitwidth, mode=mode)().to(device)
            iVecRNG = RNG(bitwidth, 1, rng)().to(device)
            iVecBS = BSGen(iVecSource, iVecRNG).to(device)

            iVecPE = ProgError(iVec, scale=1, mode=mode).to(device)
            
            if scale is True:
                if bias is False:
                    oVecPE = ProgError(oVec, scale=in_feature, mode=mode).to(device)
                elif bias is True:
                    oVecPE = ProgError(oVec, scale=in_feature+1, mode=mode).to(device)
            else:
                oVecPE = ProgError(oVec, scale=1, mode=mode).to(device)
            
            with torch.no_grad():
                idx = torch.zeros(iVecSource.size()).type(torch.long).to(device)
                start_time = time.time()
                for i in range(length):
                    iBS = iVecBS(idx + i)
                    iVecPE.Monitor(iBS)

                    oVecU = ufc(iBS)
                    oVecPE.Monitor(oVecU)
                    rmse = torch.sqrt(torch.mean(torch.mul(oVecPE()[1], oVecPE()[1])))
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
in_feature = 256
out_feature = 10000
bitwidth = 8
bias = True
linear_test(rng, in_feature, out_feature, bitwidth, bias)

# # %%
# rng = "LFSR"
# in_feature = 512
# out_feature = 10000
# bitwidth = 8
# bias = True
# linear_test(rng, in_feature, out_feature, bitwidth, bias)

# # %%
# rng = "Race"
# in_feature = 512
# out_feature = 10000
# bitwidth = 8
# bias = True
# linear_test(rng, in_feature, out_feature, bitwidth, bias)

# # %%
# rng = "SYS"
# in_feature = 512
# out_feature = 10000
# bitwidth = 8
# bias = True
# linear_test(rng, in_feature, out_feature, bitwidth, bias)

# # %%
