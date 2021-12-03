import torch
from UnarySim.stream import BinGen

def test_bingen():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    uData = torch.tensor([0.5, 0.7]).to(device)
    hwcfg = {
        "width" : 4,
        "mode" : "bipolar"
    }
    swcfg={
        "rtype" : torch.float
    }

    srcbin = BinGen(uData, hwcfg, swcfg).to(device)
    print(srcbin)
    print(srcbin())

    hwcfg["mode"] = "unipolar"
    srcbin = BinGen(uData, hwcfg, swcfg).to(device)
    print(srcbin)
    print(srcbin())

if __name__ == '__main__':
    test_bingen()
