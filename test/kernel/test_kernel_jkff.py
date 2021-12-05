import torch
from UnarySim.kernel import JKFF

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_jkff():
    swcfg = {
            "rtype" : torch.float,
            "stype" : torch.float,
            "btype" : torch.float
        }
    jkff = JKFF(swcfg).to(device)

    j = torch.tensor([[0., 0., 1., 1.]]).type(torch.float).to(device)
    k = torch.tensor([[0., 1., 0., 1.]]).type(torch.float).to(device)

    print(jkff(j,k))

    j = torch.tensor([[1., 1., 0., 0.]]).type(torch.float).to(device)
    k = torch.tensor([[1., 0., 1., 0.]]).type(torch.float).to(device)

    print(jkff(j,k))


if __name__ == '__main__':
    test_jkff()
