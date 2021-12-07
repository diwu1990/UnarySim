import torch
from UnarySim.kernel import FSULinear
from UnarySim.stream import RNG, BinGen, BSGen
from UnarySim.metric import ProgError
import matplotlib.pyplot as plt
import time
import torch.autograd.profiler as profiler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_fsulinear():
    plot_en=False

    hwcfg_input={
        "width" : 12,
        "mode" : "bipolar",
        "scale" : None,
        "depth" : 20,
        "rng" : "Sobol",
        "dimr" : 1
    }
    hwcfg={
        "width" : 12,
        "mode" : "bipolar",
        "scale" : None,
        "depth" : 20,
        "rng" : "Sobol",
        "dimr" : 1
    }
    swcfg={
        "btype" : torch.float, 
        "rtype" : torch.float, 
        "stype" : torch.float
    }

    rng = hwcfg["rng"]
    in_feature = 256
    out_feature = 10000
    bitwidth = hwcfg["width"]
    bias = True
    # modes = ["bipolar"]
    # scaled = [False]
    modes = ["bipolar", "unipolar"]
    scaled = [True, False]
    result_pe = []
    
    for mode in modes:
        for scale in scaled:
            hwcfg["mode"] = mode
            hwcfg_input["mode"] = mode
            hwcfg["scale"] = (in_feature + bias) if scale else 1
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

            ufc = FSULinear(in_feature, out_feature, bias=bias, weight_ext=fc.weight, bias_ext=fc.bias, 
                              hwcfg=hwcfg, swcfg=swcfg).to(device)

            iVec = ((torch.rand(1, in_feature)*length).round()/length).to(device)
            oVec = fc(iVec)

            iVecSource = BinGen(iVec, hwcfg, swcfg)().to(device)
            iVecRNG = RNG(hwcfg_input, swcfg)().to(device)
            iVecBS = BSGen(iVecSource, iVecRNG, hwcfg, swcfg).to(device)

            hwcfg["scale"] = 1
            iVecPE = ProgError(iVec, hwcfg).to(device)

            hwcfg["scale"] = (in_feature + bias) if scale else 1
            oVecPE = ProgError(oVec, hwcfg).to(device)
            
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
    test_fsulinear()
