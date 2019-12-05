import torch

class ShiftReg(torch.nn.Module):
    """
    this module is a shift register for int8 tensor.
    """
    def __init__(self,
                 depth=8):
        super(ShiftReg, self).__init__()
        self.depth = 8
        self.sr = torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)

    def ShiftReg_forward(self, input):
        if self.sr.shape == [self.depth, input.shape]:
            # shape does not match, do broadcasting and initialization
            pass
        
        # do shifting
                 
    def forward(self, input):
        return self.ShiftReg_forward(input)

    