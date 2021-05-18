# %%
%load_ext autoreload
%autoreload 2

# %%
import torch
from UnarySim.sw.kernel.mul import UnaryMul
from UnarySim.sw.stream.gen import RNG, SourceGen, BSGen
from UnarySim.sw.metric.metric import ProgressiveError
import matplotlib.pyplot as plt
import time
import math

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
rng = "Sobol"

# %%
col = 10000
modes = ["bipolar", "unipolar"]
static = True
bitwidth = 8
stype = torch.int8
rtype = torch.long

for mode in modes:
    if mode == "unipolar":
        input_prob = torch.rand(col).mul(2**bitwidth).round().div(2**bitwidth).to(device)
        iVec = torch.rand(col).mul(2**bitwidth).round().div(2**bitwidth).to(device)
    elif mode == "bipolar":
        input_prob = torch.rand(col).mul(2).sub(1).mul(2**bitwidth).round().div(2**bitwidth).to(device)
        iVec = torch.rand(col).mul(2).sub(1).mul(2**bitwidth).round().div(2**bitwidth).to(device)

    dut_mul = UnaryMul(bitwidth=bitwidth, mode=mode, static=static, input_prob_1=input_prob, stype=stype, rtype=rtype).to(device)

    oVec = torch.mul(iVec, input_prob).mul(2**bitwidth).round().div(2**bitwidth).to(device)

    iVecPE = ProgressiveError(iVec, mode=mode).to(device)
    oVecPE = ProgressiveError(oVec, mode=mode).to(device)

    iVecSource = SourceGen(iVec, bitwidth, mode=mode)().to(device)
    iVecRNG = RNG(bitwidth, 1, rng)().to(device)
    iVecBS = BSGen(iVecSource, iVecRNG, stype).to(device)

    with torch.no_grad():
        start_time = time.time()
        for i in range(2**bitwidth):             # unary cycle count 2^n for n-bit binary data
            iBS = iVecBS(torch.tensor([i]))      # input bit stream generation
            iVecPE.Monitor(iBS)                  # input accuracy measurement
            oVecU = dut_mul(iBS)                 # computing kernel, e.g., multiplication
            oVecPE.Monitor(oVecU)                # output accuracy measurement
        print("--- %s seconds ---" % (time.time() - start_time))
        print("input error: ", torch.min(iVecPE()[1]), torch.max(iVecPE()[1]))
        print("output error: ", torch.min(oVecPE()[1]), torch.max(oVecPE()[1]))
        result_pe = oVecPE()[1].cpu().numpy()
        print("RMSE", math.sqrt(sum(result_pe**2)/len(result_pe)))
        print("bias", sum(result_pe)/len(result_pe))
        fig = plt.hist(result_pe, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram for final output error")
        plt.show()

# %%
