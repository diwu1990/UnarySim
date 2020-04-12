import torch
from UnarySim.sw.kernel.shiftreg import ShiftReg

class UnaryAbs(torch.nn.Module):
    """
    this module is to calculate the bipolar absolute value unary data, based on non-scaled unipolar unary addition.
    only works for rate-coded data.
    """
    def __init__(self, 
                 depth=8, 
                 shiftreg=False, 
                 btype=torch.float, 
                 stype=torch.float):
        super(UnaryAbs, self).__init__()
        self.depth = depth
        self.sr = shiftreg
        self.stype = stype
        self.btype = btype
        if shiftreg is True:
            assert depth <= 127, "When using shift register implementation, buffer depth should be less than 127."
            self.shiftreg = ShiftReg(depth, self.stype)
            self.depth_half = torch.nn.Parameter(torch.zeros(1).fill_(depth/2).type(btype), requires_grad=False)
            self.sr_cnt = torch.nn.Parameter(torch.zeros(1).type(self.stype), requires_grad=False)
        else:
            self.buf_max = torch.nn.Parameter(torch.zeros(1).fill_(2**depth - 1).type(btype), requires_grad=False)
            self.buf_half = torch.nn.Parameter(torch.zeros(1).fill_(2**(depth - 1)).type(btype), requires_grad=False)
            self.acc = torch.nn.Parameter(torch.zeros(1).fill_(2**(depth - 1)).type(btype), requires_grad=False)
    
    def forward(self, input):
        if self.sr is True:
            # update shiftreg based on input
            _, self.sr_cnt.data = self.shiftreg(input)
            half_prob_flag = torch.ge(self.sr_cnt, self.depth_half).type(torch.int8)
        else:
            # update the accumulator based on input
            self.acc.data = self.acc.add(input.mul(2).sub(1).type(self.btype)).clamp(0, self.buf_max.item())
            half_prob_flag = torch.ge(self.acc, self.buf_half).type(torch.int8)
        sign = 1 - half_prob_flag
        input_int8 = input.type(torch.int8)
        output = (half_prob_flag & input_int8) | (sign & (1 - input_int8))
        return sign.type(self.stype), output.type(self.stype)