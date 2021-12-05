import torch
from UnarySim.kernel.shiftreg import ShiftReg

class FSUReLU(torch.nn.Module):
    """
    Unary ReLU activation by comparing the bitstream value in a counter with bipolar 0 
    Bitstream should always bipolar and rate coded
    """
    def __init__(
        self, 
        hwcfg={
            "depth" : 6
        },
        swcfg={
            "btype" : torch.float, 
            "stype" : torch.float
        }):
        super(FSUReLU, self).__init__()
        self.hwcfg = {}
        self.hwcfg["depth"] = hwcfg["depth"]

        self.swcfg = {}
        self.swcfg["btype"] = swcfg["btype"]
        self.swcfg["stype"] = swcfg["stype"]

        self.depth = hwcfg["depth"]
        self.stype = swcfg["btype"]
        self.btype = swcfg["stype"]

        self.buf_max = 2**self.depth - 1
        self.buf_half = 2**(self.depth - 1)
        self.acc = torch.nn.Parameter(torch.zeros(1).fill_(2**(self.depth - 1)).type(self.btype), requires_grad=False)
    
    def forward(self, input):
        # check whether acc is larger than or equal to half.
        half_prob_flag = torch.ge(self.acc, self.buf_half).type(torch.int8)
        # only when input is 0 and flag is 1, output 0; otherwise 1
        output = input.type(torch.int8) | (1 - half_prob_flag)
        # update the accumulator based on output, thus acc update is after output generation
        self.acc.data = self.acc.add(output.mul(2).sub(1).type(self.btype)).clamp(0, self.buf_max)
        return output.type(self.stype)


class HUBReLU(torch.nn.Hardtanh):
    """
    clip the input when it is larger than 1.
    """
    def __init__(self, scale=1., inplace: bool = False):
        super(HUBReLU, self).__init__(0., scale, inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
        