import torch
from UnarySim.kernel import ShiftReg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_shiftreg():
    hwcfg = {
            "width" : 12,
            "mode" : "bipolar",
            "dim" : 0,
            "rng" : "sobol",
            "scale" : 1,
            "depth" : 4,
            "entry" : 4
        }
    swcfg = {
            "rtype" : torch.float,
            "stype" : torch.float,
            "btype" : torch.float
        }
    sr = ShiftReg(hwcfg, swcfg).to(device)

    a = torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0]]).type(swcfg["stype"]).to(device)

    oBit, cnt = sr(a, mask=torch.tensor([[1, 1], [1, 1], [0, 0], [0, 0]]).to(device))
    print(oBit, cnt)
    print(sr.sr)


if __name__ == '__main__':
    test_shiftreg()
