import torch
from UnarySim.stream.gen import RNG, BinGen, BSGen
from UnarySim.metric.metric import ProgError
import matplotlib.pyplot as plt
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_bsgen():
    hwcfg = {
            "width" : 8,
            "mode" : "bipolar",
            "dimr" : 1,
            "rng" : "sobol",
            "scale" : 1
        }
    swcfg = {
            "rtype" : torch.float,
            "stype" : torch.float,
            "btype" : torch.float
        }
    bitwidth = hwcfg["width"]
    mode = hwcfg["mode"]

    col = 100000

    result_pe_cycle = []
    if mode == "unipolar":
        iVec = torch.rand(1, col).mul(2**bitwidth).round().div(2**bitwidth).to(device)
    elif mode == "bipolar":
        iVec = torch.rand(1, col).mul(2).sub(1).mul(2**bitwidth).round().div(2**bitwidth).to(device)

    iVecSource = BinGen(iVec, hwcfg, swcfg)().to(device)

    iVecRNG = RNG(hwcfg, swcfg)().to(device)
    iVecBS = BSGen(iVecSource, iVecRNG, swcfg).to(device)

    iVecPE = ProgError(iVec, hwcfg).to(device)
    with torch.no_grad():
        idx = torch.zeros(iVecSource.size()).type(torch.long).to(device)
        start_time = time.time()
        for i in range(2**bitwidth):
            iBS = iVecBS(idx + i)
            iVecPE.Monitor(iBS)
            result_pe_cycle.append(1-torch.sqrt(torch.sum(torch.mul(iVecPE()[1][0], iVecPE()[1][0]))/col).item())
        print("--- %s seconds ---" % (time.time() - start_time))
        print("input error: ", "min:", torch.min(iVecPE()[1]).item(), "max:", torch.max(iVecPE()[1]).item(), 
                                "mean:", torch.mean(iVecPE()[1]).item(), "rmse:", torch.sqrt(torch.mean(torch.square(iVecPE()[1]))).item())
        # result_pe = iVecPE()[1][0].cpu().numpy()
        # plt.figure(figsize=(3,1.5))
        # fig = plt.hist(result_pe, bins='auto')  # arguments are passed to np.histogram
        # plt.title("error distribution for data: "+hwcfg["mode"])
        # plt.show()
        # plt.figure(figsize=(3,1.5))
        # fig = plt.plot(result_pe_cycle)  # arguments are passed to np.histogram
        # plt.title("progressive accuracy for data: "+hwcfg["mode"])
        # plt.show()


if __name__ == '__main__':
    test_bsgen()
