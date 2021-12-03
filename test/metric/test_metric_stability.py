import torch
from UnarySim.metric.metric import Stability

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_stability():
    hwcfg = {
            "mode" : "bipolar",
            "scale" : 1,
            "threshold" : 0.05,
        }

    input = torch.tensor([-0.5,0]).to(device)

    stb = Stability(input, hwcfg).to(device)

    a = torch.tensor([1,0]).type(torch.int8).to(device)
    stb.Monitor(a)

    a = torch.tensor([0,1]).type(torch.int8).to(device)
    stb.Monitor(a)

    a = torch.tensor([1,0]).type(torch.int8).to(device)
    stb.Monitor(a)

    a = torch.tensor([0,1]).type(torch.int8).to(device)
    stb.Monitor(a)

    a = torch.tensor([1,0]).type(torch.int8).to(device)
    stb.Monitor(a)

    a = torch.tensor([0,1]).type(torch.int8).to(device)
    stb.Monitor(a)

    a = torch.tensor([1,0]).type(torch.int8).to(device)
    stb.Monitor(a)

    a = torch.tensor([0,1]).type(torch.int8).to(device)
    stb.Monitor(a)

    a = torch.tensor([1,0]).type(torch.int8).to(device)
    stb.Monitor(a)

    a = torch.tensor([0,1]).type(torch.int8).to(device)
    stb.Monitor(a)

    print(stb.pe)
    print(stb.cycle_to_stable)
    print(stb.cycle)
    print(stb.threshold)
    print(stb())


if __name__ == '__main__':
    test_stability()
