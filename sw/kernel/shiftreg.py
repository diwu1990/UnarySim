import torch

class ShiftReg(torch.nn.Module):
    """
    this module is a shift register for int8 tensor.
    it takes in unary bits and output shifter register value, as well as the total number of 1s in current shift register.
    """
    def __init__(self,
                 depth=8,
                 bstype=torch.float):
        super(ShiftReg, self).__init__()
        self.depth = 8
        self.sr = torch.nn.Parameter(torch.zeros(1).type(bstype), requires_grad=False)
        self.init = True
        self.bstype = bstype

    def ShiftReg_forward(self, input):
        if self.init is True:
            new_shape = [1 for _ in range(len(input.shape))]
            new_shape.insert(0, self.depth)
            self.sr.data = input.repeat(new_shape)
            for i in range(self.depth):
                self.sr[i].fill_(i%2)
            self.init = False
        
        # do shifting
        # output
        out = self.sr[0]
        # sum in current shift register
        cnt = torch.sum(self.sr, 0)
        self.sr.data = torch.roll(self.sr, -1, 0)
        self.sr.data[self.depth-1] = input.clone().detach()
        return out, cnt

    def forward(self, input):
        return self.ShiftReg_forward(input)

    