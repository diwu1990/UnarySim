import torch

class FSUSignAbs(torch.nn.Module):
    """
    This module 
    1) calculates the absolute value of bipolar bitstreams.
    2) works for rate coding only.
    3) records the entire bistream history with a counter to retreive the sign.
    """
    def __init__(
        self, 
        hwcfg={
            "depth" : 3 
        }, 
        swcfg={
            "btype" : torch.float, 
            "stype" : torch.float
        }):
        super(FSUSignAbs, self).__init__()
        self.depth = hwcfg["depth"]
        self.stype = swcfg["stype"]
        self.btype = swcfg["btype"]
        self.buf_max = torch.nn.Parameter(torch.zeros(1).fill_(2**self.depth - 1).type(self.btype), requires_grad=False)
        self.buf_half = torch.nn.Parameter(torch.zeros(1).fill_(2**(self.depth - 1)).type(self.btype), requires_grad=False)
        self.acc = torch.nn.Parameter(torch.zeros(1).fill_(2**(self.depth - 1)).type(self.btype), requires_grad=False)
    
    def forward(self, input):
        # update the accumulator based on input: +1 for input 1; -1 for input 0
        self.acc.data = self.acc.add(input.mul(2).sub(1).type(self.btype)).clamp(0, self.buf_max.item())
        sign = torch.lt(self.acc, self.buf_half).type(torch.int8)
        input_int8 = input.type(torch.int8)
        output = sign ^ input_int8
        return sign.type(self.stype), output.type(self.stype)
    
