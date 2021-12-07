
import torch
from UnarySim.stream import RNG, BinGen, BSGen
from UnarySim.stream import Bi2Uni
from UnarySim.metric import ProgError
import matplotlib.pyplot as plt
import time



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_bi2uni():
    hwcfg = {
            "width" : 8,
            "mode" : "bipolar",
            "dimr" : 1,
            "rng" : "sobol",
            "scale" : 1,
            "depth" : 3
        }
    swcfg = {
            "rtype" : torch.float,
            "stype" : torch.float,
            "btype" : torch.float
        }
    bitwidth = hwcfg["width"]
    mode = hwcfg["mode"]

    rng = "Sobol"

    in_dim = 1024
    bitwidth = 8
    in_mode = "bipolar"
    out_mode = "unipolar"
    stype = torch.float
    btype = torch.float
    rtype = torch.float

    uBi2Uni = Bi2Uni(hwcfg, swcfg).to(device)

    iVec = ((torch.rand(in_dim)*(2**bitwidth)).round()/(2**bitwidth)).to(device)
    start_time = time.time()
    oVec = iVec.type(torch.float)
    print("--- %s seconds ---" % (((time.time() - start_time))*2**bitwidth))

    print("input", iVec)
    print("real output", oVec)
    
    hwcfg["mode"] = "bipolar"
    iVecSource = BinGen(iVec, hwcfg, swcfg)().to(device)

    iVecRNG = RNG(hwcfg, swcfg)().to(device)
    iVecBS = BSGen(iVecSource, iVecRNG, hwcfg, swcfg).to(device)

    iVecPE = ProgError(iVec, hwcfg).to(device)
    hwcfg["mode"] = "unipolar"
    oVecPE = ProgError(oVec, hwcfg).to(device)

    hwcfg["mode"] = "bipolar"

    with torch.no_grad():
        idx = torch.zeros(iVecSource.size()).type(torch.long).to(device)
        start_time = time.time()
        for i in range((2**bitwidth)):
            iBS = iVecBS(idx + i)
            iVecPE.Monitor(iBS)

            oVecU = uBi2Uni(iBS)
            oVecPE.Monitor(oVecU)
        print("--- %s seconds ---" % (time.time() - start_time))
        print("final input error: ", min(iVecPE()[1]), max(iVecPE()[1]))
        print("final output error:", min(oVecPE()[1]), max(oVecPE()[1]))
        print("final output pp:", oVecPE()[0].data)
        print("final output pe:", oVecPE()[1].data)
        print("final output mean error:", oVecPE()[1].mean())
        
        result_pe = oVecPE()[1].cpu().numpy()


    # fig = plt.hist(result_pe, bins='auto')  # arguments are passed to np.histogram
    # plt.title("Histogram for final output error")
    # plt.show()


    print(result_pe)
    print(result_pe.argmin(), result_pe.argmax())
    print(result_pe[result_pe.argmin()], result_pe[result_pe.argmax()])
    print(iVec[result_pe.argmin()], iVec[result_pe.argmax()])


if __name__ == '__main__':
    test_bi2uni()
