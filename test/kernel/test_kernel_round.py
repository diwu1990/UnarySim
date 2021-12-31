import torch
from UnarySim.kernel import Round

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_round():
    dtype = torch.float32
    a1 = torch.rand(4, requires_grad=True, dtype=dtype)
    print(a1)
    b1 = torch.round(a1) * 8
    b1.retain_grad()
    print(b1)
    c1 = torch.sum(b1*b1)
    print(c1)
    c1.backward()
    print(a1.grad)
    print(b1.grad)
    print()

    a2 = a1.detach().clone()
    a2 = a2.requires_grad_()
    b2 = Round(0, 7)(a2) * 8
    b2.retain_grad()
    print(b2)
    c2 = torch.sum(b2*b2)
    print(c2)
    c2.backward()
    print(a2.grad)
    print(b2.grad)
    print()

    a3 = a1.detach().clone()
    a3 = a3.requires_grad_()
    b3 = a3 * 8
    b3.retain_grad()
    print(b3)
    c3 = torch.sum(b3*b3)
    print(c3)
    c3.backward()
    print(a3.grad)
    print(b3.grad)
    print()


if __name__ == '__main__':
    test_round()

