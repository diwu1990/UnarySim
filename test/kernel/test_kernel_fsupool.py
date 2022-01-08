import torch
from UnarySim.stream import RNG, BinGen, BSGen
from UnarySim.metric import ProgError
from UnarySim.kernel.pool import FSUAvgPool2d
import matplotlib.pyplot as plt
import time
import math


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_fsupool():
    width = 8
    format = "bfloat16"

    if format in ["fxp", "float32"]:
        dtype = torch.float32
    elif format in ["bfloat16"]:
        dtype = torch.bfloat16
    elif format in ["float16"]:
        dtype = torch.float16

    hwcfg = {
            "width" : width,
            "mode" : None,
            "dimr" : 1,
            "dima" : 0,
            "rng" : "sobol",
            "depth" : 12,
            "scale" : None,
            "entry" : None,
            "leak" : 0.5,
            "widthg" : 1.25
        }
    swcfg = {
            "rtype" : dtype,
            "stype" : dtype,
            "btype" : torch.float
        }
    bitwidth = hwcfg["width"]

    batch = 32
    channel = 3
    height = 64
    width = 64
    modes = ["bipolar", "unipolar"]

    for mode in modes:
        if mode == "unipolar":
            iVec = torch.rand((batch, channel, height, width), dtype=dtype).mul(2**bitwidth).round().div(2**bitwidth).to(device)
        elif mode == "bipolar":
            iVec = torch.rand((batch, channel, height, width), dtype=dtype).mul(2).sub(1).mul(2**bitwidth).round().div(2**bitwidth).to(device)
        hwcfg["mode"] = mode

        dut_pool = FSUAvgPool2d(2, hwcfg=hwcfg, swcfg=swcfg).to(device)

        oVec = torch.nn.AvgPool2d(2)(iVec).to(device)

        hwcfg["scale"] = 1
        iVecPE = ProgError(iVec, hwcfg).to(device)
        oVecPE = ProgError(oVec, hwcfg).to(device)
        hwcfg["scale"] = None

        iVecSource = BinGen(iVec, hwcfg, swcfg)().to(device)
        iVecRNG = RNG(hwcfg, swcfg)().to(device)
        iVecBS = BSGen(iVecSource, iVecRNG, swcfg).to(device)

        with torch.no_grad():
            start_time = time.time()
            for i in range(2**bitwidth):             # unary cycle count 2^n for n-bit binary data
                iBS = iVecBS(torch.tensor([i]))      # input bit stream generation
                iVecPE.Monitor(iBS)                  # input accuracy measurement
                oVecU = dut_pool(iBS)                 # computing kernel, e.g., multiplication
                oVecPE.Monitor(oVecU)                # output accuracy measurement
                rmse = torch.sqrt(torch.mean(torch.mul(oVecPE()[1], oVecPE()[1])))
            print("--- %s seconds ---" % (time.time() - start_time))
            print(dut_pool.hwcfg)
            print(dut_pool.swcfg)
            print("Data: "+mode)
            print("input error:  ", "min: ", torch.min(iVecPE()[1]).item(), "max: ", torch.max(iVecPE()[1]).item())
            print("output error: ", "min: ", torch.min(oVecPE()[1]).item(), "max: ", torch.max(oVecPE()[1]).item(), "RMSE: ", rmse.item())
            print()

if __name__ == '__main__':
    test_fsupool()

