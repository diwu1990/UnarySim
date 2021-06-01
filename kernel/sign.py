import torch
from UnarySim.kernel.shiftreg import ShiftReg

class FSUSign(torch.nn.Module):
    """
    return the sign of unary bipolar rate-coded data
    0 for non-negative, and 1 for negative
    """
    def __init__(self, 
                 depth=3, 
                 shiftreg=False, 
                 btype=torch.float, 
                 stype=torch.float):
        super(FSUSign, self).__init__()
        self.depth = depth
        self.sr = shiftreg
        self.stype = stype
        self.btype = btype
        # self.sign_sr = ShiftReg(8, torch.int8)
        # self.sign_cnt = torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)
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
        sign_temp = 1 - half_prob_flag
        # _, self.sign_cnt.data = self.sign_sr(sign_temp)
        # sign = torch.gt(self.sign_cnt, 4).type(torch.int8)
        sign = sign_temp
        return sign.type(self.stype)
    