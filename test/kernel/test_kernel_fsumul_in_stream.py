import torch
from UnarySim.kernel import FSUMul
from UnarySim.stream import RNG, BinGen, BSGen
from UnarySim.metric import ProgError
import matplotlib.pyplot as plt
import time
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_fsumul_in_stream():
    bitwidth = 12
    depth = 4
    hwcfg = {
            "width" : bitwidth,
            "mode" : "bipolar",
            "dimr" : 1,
            "dima" : 0,
            "rng" : "sobol",
            "scale" : 1,
            "depth" : 10,
            "entry" : None,
            "static" : False
        }
    swcfg = {
            "rtype" : torch.float,
            "stype" : torch.float,
            "btype" : torch.float
        }

    col = 100
    modes = ["bipolar", "unipolar"]

    for mode in modes:
        if mode == "unipolar":
            input_prob_0 = torch.rand(col).mul(2**bitwidth).round().div(2**bitwidth).to(device)
            input_prob_1 = torch.rand(col).mul(2**bitwidth).round().div(2**bitwidth).to(device)
        elif mode == "bipolar":
            input_prob_0 = torch.rand(col).mul(2).sub(1).mul(2**bitwidth).round().div(2**bitwidth).to(device)
            input_prob_1 = torch.rand(col).mul(2).sub(1).mul(2**bitwidth).round().div(2**bitwidth).to(device)

        hwcfg["mode"] = mode

        hwcfg["width"] = depth
        dut_mul = FSUMul(None, hwcfg, swcfg).to(device)
        hwcfg["width"] = bitwidth

        oVec = torch.mul(input_prob_0, input_prob_1).mul(2**bitwidth).round().div(2**bitwidth).to(device)

        prob_0_PE = ProgError(input_prob_0, hwcfg).to(device)
        prob_1_PE = ProgError(input_prob_1, hwcfg).to(device)

        oVecPE = ProgError(oVec, hwcfg).to(device)

        prob_0_Source = BinGen(input_prob_0, hwcfg, swcfg)().to(device)
        prob_1_Source = BinGen(input_prob_1, hwcfg, swcfg)().to(device)

        iVecRNG0 = RNG(hwcfg, swcfg)().to(device)
        iVecRNG1 = RNG(hwcfg, swcfg)().to(device)
        prob_0_BS = BSGen(prob_0_Source, iVecRNG0, swcfg).to(device)
        prob_1_BS = BSGen(prob_1_Source, iVecRNG1, swcfg).to(device)

        with torch.no_grad():
            start_time = time.time()
            idx = torch.zeros(input_prob_0.size()).type(torch.long).to(device)
            for i in range(2**bitwidth):
                #print(i)
                iBS_0 = prob_0_BS(idx + i)
                iBS_1 = prob_1_BS(idx + i)

                prob_0_PE.Monitor(iBS_0)
                prob_1_PE.Monitor(iBS_1)

                oVecU = dut_mul(iBS_0, iBS_1)   
                oVecPE.Monitor(oVecU)
            print("--- %s seconds ---" % (time.time() - start_time))
            print(mode)
            print("input 0 error: ", "min:", torch.min(prob_0_PE()[1]), "max:", torch.max(prob_0_PE()[1]))
            print("input 1 error: ", "min:", torch.min(prob_1_PE()[1]), "max:", torch.max(prob_1_PE()[1]))

            print("output error: ", "min:", torch.min(oVecPE()[1]), "max:", torch.max(oVecPE()[1]), "rmse:", torch.sqrt(torch.mean(torch.mul(oVecPE()[1], oVecPE()[1]))), "bias:", torch.mean(oVecPE()[1]))
            # result_pe = oVecPE()[1].cpu().numpy().flatten()
            # fig = plt.hist(result_pe, bins='auto')  # arguments are passed to np.histogram
            # plt.title("Histogram for final output error")
            # plt.show()



if __name__ == '__main__':
    test_fsumul_in_stream()
