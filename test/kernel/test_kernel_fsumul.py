import torch
from UnarySim.kernel.mul import FSUMul
from UnarySim.stream.gen import RNG, BinGen, BSGen
from UnarySim.metric.metric import ProgError
import matplotlib.pyplot as plt
import time
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_fsumul():
    hwcfg = {
            "width" : 8,
            "mode" : "bipolar",
            "dimr" : 1,
            "dima" : 0,
            "rng" : "sobol",
            "scale" : 1,
            "depth" : 10,
            "entry" : None,
            "static" : True
        }
    swcfg = {
            "rtype" : torch.float,
            "stype" : torch.float,
            "btype" : torch.float
        }
    bitwidth = hwcfg["width"]

    col = 100
    modes = ["bipolar", "unipolar"]

    for mode in modes:
        if mode == "unipolar":
            input_prob = torch.rand(col).mul(2**bitwidth).round().div(2**bitwidth).to(device)
            iVec = torch.rand(col).mul(2**bitwidth).round().div(2**bitwidth).to(device)
        elif mode == "bipolar":
            input_prob = torch.rand(col).mul(2).sub(1).mul(2**bitwidth).round().div(2**bitwidth).to(device)
            iVec = torch.rand(col).mul(2).sub(1).mul(2**bitwidth).round().div(2**bitwidth).to(device)

        hwcfg["mode"] = mode

        dut_mul = FSUMul(input_prob, hwcfg, swcfg).to(device)

        oVec = torch.mul(iVec, input_prob).mul(2**bitwidth).round().div(2**bitwidth).to(device)

        iVecPE = ProgError(iVec, hwcfg).to(device)
        oVecPE = ProgError(oVec, hwcfg).to(device)

        iVecSource = BinGen(iVec, hwcfg, swcfg)().to(device)
        iVecRNG = RNG(hwcfg, swcfg)().to(device)
        iVecBS = BSGen(iVecSource, iVecRNG, swcfg).to(device)

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
            # fig = plt.hist(result_pe, bins='auto')  # arguments are passed to np.histogram
            # plt.title("Histogram for final output error")
            # plt.show()


if __name__ == '__main__':
    test_fsumul()
