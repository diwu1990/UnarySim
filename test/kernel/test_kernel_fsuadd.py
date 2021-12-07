import torch
from UnarySim.kernel import FSUAdd
from UnarySim.stream import RNG, BinGen, BSGen
from UnarySim.metric import ProgError
import matplotlib.pyplot as plt
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_fsuadd():
    hwcfg = {
            "width" : 12,
            "mode" : "bipolar",
            "dimr" : 1,
            "dima" : 0,
            "rng" : "sobol",
            "scale" : 1,
            "depth" : 20,
            "entry" : None
        }
    swcfg = {
            "rtype" : torch.float,
            "stype" : torch.float,
            "btype" : torch.float
        }
    bitwidth = hwcfg["width"]
    rng = hwcfg["rng"]

    plot_en=False
    modes = ["bipolar", "unipolar"]
    size = [128, 256, 512]

    scaled = [True, False]
    result_pe = []

    for mode in modes:
        for scale in scaled:
            run_time = 0
            acc_dim = hwcfg["dima"]
            scale_mod = size[acc_dim]
            result_pe_cycle = []
            hwcfg["mode"] = mode
            hwcfg["scale"] = scale_mod if scale else 1
            uadd = FSUAdd(hwcfg, swcfg).to(device)

            if mode == "unipolar":
                iVec = torch.rand(size).mul(2**bitwidth).round().div(2**bitwidth).to(device)
            elif mode == "bipolar":
                iVec = torch.rand(size).mul(2).sub(1).mul(2**bitwidth).round().div(2**bitwidth).to(device)
            
            oVec = torch.sum(iVec, acc_dim).to(device)

            iVecSource = BinGen(iVec, hwcfg, swcfg)().to(device)
            iVecRNG = RNG(hwcfg, swcfg)().to(device)
            iVecBS = BSGen(iVecSource, iVecRNG, swcfg).to(device)
            hwcfg["scale"] = 1
            iVecPE = ProgError(iVec, hwcfg).to(device)
            print("iVecPE cfg", iVecPE.hwcfg)
            hwcfg["scale"] = scale_mod if scale else 1
            oVecPE = ProgError(oVec, hwcfg).to(device)
            print("oVecPE cfg", oVecPE.hwcfg)

            with torch.no_grad():
                idx = torch.zeros(iVecSource.size()).type(torch.long).to(device)
                for i in range(2**bitwidth):
                    iBS = iVecBS(idx + i)
                    iVecPE.Monitor(iBS)
                    
                    start_time = time.time()
                    oVecU = uadd(iBS)
                    run_time = time.time() - start_time + run_time

                    if i == 0:
                        print("uadd cfg", uadd.hwcfg)
                    
                    oVecPE.Monitor(oVecU)
                    rmse = torch.sqrt(torch.mean(torch.mul(oVecPE()[1], oVecPE()[1])))
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
    test_fsuadd()
