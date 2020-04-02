import torch

class JKFF(torch.nn.Module):
    """
    this module is a shift register for int8 tensor.
    it takes in unary bits and output shifter register value, as well as the total number of 1s in current shift register.
    """
    def __init__(self,
                 bstype=torch.float):
        super(JKFF, self).__init__()
        self.jkff = torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)
        self.bstype = bstype

    def forward(self, J, K):
        j0 = torch.eq(J, 0).type(torch.int8)
        j1 = 1 - j0
        k0 = torch.eq(K, 0).type(torch.int8)
        k1 = 1 - k0
        
        j0k0 = j0 & k0
        j1k0 = j1 & k0
        j0k1 = j0 & k1
        j1k1 = j1 & k1
        
        self.jkff.data = j0k0 * self.jkff + j1k0 * torch.ones_like(J, dtype=torch.int8) + j0k1 * torch.zeros_like(J, dtype=torch.int8) + j1k1 * (1 - self.jkff)
        return self.jkff.type(self.bstype)

    