# %%
import torch
from UnarySim.sw.kernel.add import GainesAdd
from UnarySim.sw.stream.gen import RNG, SourceGen, BSGen
from UnarySim.sw.metric.metric import ProgressiveError
import matplotlib.pyplot as plt
import time, math

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
def add_test(rng="Sobol", row=128, col=10000, bitwidth=8):
    modes = ["unipolar", "bipolar"]

    scaled = [True, False]
    result_pe = []
    stype = torch.float
    btype = torch.float
    rtype = torch.float
    
    for mode in modes:
        for scale in scaled:
            acc_dim = 0
            result_pe_cycle = []
            uadd = GainesAdd(mode=mode, scaled=scale, acc_dim=acc_dim, 
                             rng="Sobol", rng_dim=5, rng_width=int(math.log2(row)), 
                             stype=stype, rtype=rtype).to(device)

            if mode == "unipolar":
                iVec = torch.rand(row, col).mul(2**bitwidth).round().div(2**bitwidth).to(device)
            elif mode == "bipolar":
                iVec = torch.rand(row, col).mul(2).sub(1).mul(2**bitwidth).round().div(2**bitwidth).to(device)
            
            oVec = torch.sum(iVec, acc_dim).to(device)

            iVecSource = SourceGen(iVec, bitwidth=bitwidth, mode=mode, rtype=rtype)().to(device)

            iVecRNG = RNG(bitwidth, 1, rng, rtype=rtype)().to(device)
            iVecBS = BSGen(iVecSource, iVecRNG, stype=stype).to(device)

            iVecPE = ProgressiveError(iVec, scale=1, mode=mode).to(device)
            
            if scale is True:
                if acc_dim == 0:
                    oVecPE = ProgressiveError(oVec, scale=row, mode=mode).to(device)
                elif acc_dim ==1:
                    oVecPE = ProgressiveError(oVec, scale=col, mode=mode).to(device)
            else:
                oVecPE = ProgressiveError(oVec, scale=1, mode=mode).to(device)
            
            
            with torch.no_grad():
                idx = torch.zeros(iVecSource.size()).type(torch.long).to(device)
                start_time = time.time()
                for i in range(2**bitwidth):
                    iBS = iVecBS(idx + i)
                    iVecPE.Monitor(iBS)
                    
                    oVecU = uadd(iBS)
                    oVecPE.Monitor(oVecU)
                    result_pe_cycle.append(1-torch.sqrt(torch.sum(torch.mul(oVecPE()[1], oVecPE()[1]))/col).item())
                print("--- %s seconds ---" % (time.time() - start_time))
                print("input error: ", "min:", torch.min(iVecPE()[1]).item(), "max:", torch.max(iVecPE()[1]).item())
                print("output error: ", "min:", torch.min(oVecPE()[1]).item(), "max:", torch.max(oVecPE()[1]).item())
                result_pe = oVecPE()[1].cpu().numpy()
                print("error distribution=========>")
                plt.figure(figsize=(3,1.5))
                fig = plt.hist(result_pe, bins='auto')  # arguments are passed to np.histogram
                plt.title("data: "+mode+", scaled: "+str(scale))
                plt.show()
                print("progressive accuracy=========>")
                plt.figure(figsize=(3,1.5))
                fig = plt.plot(result_pe_cycle)  # arguments are passed to np.histogram
                plt.title("data: "+mode+", scaled: "+str(scale))
                plt.show()

# %%
rng = "Sobol"
row = 128
col = 10000
bitwidth = 8
add_test(rng, row, col, bitwidth)

# %%
