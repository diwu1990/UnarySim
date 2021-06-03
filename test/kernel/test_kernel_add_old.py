# %%
import torch
from UnarySim.kernel.add import FSUAdd_old
from UnarySim.stream.gen import RNG, SourceGen, BSGen
from UnarySim.metric.metric import ProgressiveError
import matplotlib.pyplot as plt
import time

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
def add_test(rng="Sobol", row=128, col=10000, bitwidth=8, plot_en=False):
    modes = ["bipolar", "unipolar"]

    scaled = [True, False]
    result_pe = []
    stype = torch.float
    btype = torch.float
    rtype = torch.float

    for mode in modes:
        for scale in scaled:
            run_time = 0
            acc_dim = 0
            result_pe_cycle = []
            uadd = FSUAdd_old(mode=mode, scaled=scale, acc_dim=acc_dim).to(device)

            if mode == "unipolar":
                iVec = torch.rand(row, col).mul(2**bitwidth).round().div(2**bitwidth).to(device)
            elif mode == "bipolar":
                iVec = torch.rand(row, col).mul(2).sub(1).mul(2**bitwidth).round().div(2**bitwidth).to(device)
            
            oVec = torch.sum(iVec, acc_dim).to(device)

            iVecSource = SourceGen(iVec, bitwidth=bitwidth, mode=mode, rtype=rtype)().to(device)

            iVecRNG = RNG(bitwidth, 1, rng, rtype)().to(device)
            iVecBS = BSGen(iVecSource, iVecRNG, stype).to(device)

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
                for i in range(2**bitwidth):
                    iBS = iVecBS(idx + i)
                    iVecPE.Monitor(iBS)
                    
                    start_time = time.time()
                    oVecU = uadd(iBS)
                    run_time = time.time() - start_time + run_time
                    
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
row = 128
col = 10000
bitwidth = 12
add_test(rng, row, col, bitwidth)

# # %%
# rng = "Race"
# row = 128
# col = 10000
# add_test(rng, row, col)

# # %%
# rng = "LFSR"
# row = 128
# col = 10000
# add_test(rng, row, col)

# # %%
# rng = "SYS"
# row = 128
# col = 10000
# add_test(rng, row, col)

# # %%