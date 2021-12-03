import torch
from UnarySim.metric.metric import Correlation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_correlation():
    corr = Correlation().to(device)

    a = torch.tensor([0]).type(torch.int8).to(device)
    b = torch.tensor([0]).type(torch.int8).to(device)
    corr.Monitor(a,b)

    a = torch.tensor([0]).type(torch.int8).to(device)
    b = torch.tensor([1]).type(torch.int8).to(device)
    corr.Monitor(a,b)

    a = torch.tensor([1]).type(torch.int8).to(device)
    b = torch.tensor([0]).type(torch.int8).to(device)
    corr.Monitor(a,b)

    a = torch.tensor([1]).type(torch.int8).to(device)
    b = torch.tensor([1]).type(torch.int8).to(device)
    corr.Monitor(a,b)

    print("d", corr.paired_00_d)
    print("c", corr.paired_01_c)
    print("b", corr.paired_10_b)
    print("a", corr.paired_11_a)

    print("SCC: ", corr())


if __name__ == '__main__':
    test_correlation()
