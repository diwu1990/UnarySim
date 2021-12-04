import torch
from UnarySim.stream import SkewedSync


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_skewedsync():
    hwcfg = {
            "depth" : 3,
        }
    swcfg = {
            "rtype" : torch.float,
            "stype" : torch.float,
            "btype" : torch.float
        }

    ss = SkewedSync(hwcfg, swcfg).to(device)

    a = torch.tensor([[0, 0]]).to(device)
    b = torch.tensor([[1, 1]]).to(device)
    print(ss(a,b))
    print(ss.cnt)

    a = torch.tensor([[1, 1]]).to(device)
    b = torch.tensor([[0, 0]]).to(device)
    print(ss(a,b))
    print(ss.cnt)

    a = torch.tensor([[1, 1]]).to(device)
    b = torch.tensor([[1, 1]]).to(device)
    print(ss(a,b))
    print(ss.cnt)

    a = torch.tensor([[0, 0]]).to(device)
    b = torch.tensor([[0, 0]]).to(device)
    print(ss(a,b))
    print(ss.cnt)

    a = torch.tensor([[0, 0]]).to(device)
    b = torch.tensor([[1, 1]]).to(device)
    print(ss(a,b))
    print(ss.cnt)


if __name__ == '__main__':
    test_skewedsync()
