import torch
from UnarySim.sw.kernel.shiftreg import ShiftReg

class UnarySign(torch.nn.Module):
    """
    return the sign of unary bipolar rate-coded data
    0 for non-negative, and 1 for negative
    """
    def __init__(self, depth=4, shiftreg=False, bstype=torch.float, buftype=torch.float):
        super(UnarySign, self).__init__()
        self.depth = depth
        self.depth_half = torch.nn.Parameter(torch.zeros(1).fill_(depth/2).type(buftype), requires_grad=False)
        self.sr = shiftreg
        self.bstype = bstype
        self.buftype = buftype
        if shiftreg is True:
            assert depth <= 127, "When using shift register implementation, buffer depth should be less than 127."
            self.shiftreg = ShiftReg(depth, self.bstype)
            self.sr_cnt = torch.nn.Parameter(torch.zeros(1).type(self.bstype), requires_grad=False)
        else:
            self.buf_max = torch.nn.Parameter(torch.zeros(1).fill_(2**depth - 1).type(buftype), requires_grad=False)
            self.buf_half = torch.nn.Parameter(torch.zeros(1).fill_(2**(depth - 1)).type(buftype), requires_grad=False)
            self.acc = torch.nn.Parameter(torch.zeros(1).fill_(2**(depth - 1)).type(buftype), requires_grad=False)

    def UnarySign_forward_rc(self, input):
        # check whether acc is less than half, i.e., bipolar zero.
        output = torch.lt(self.acc, self.buf_half).type(self.bstype)
        # update the accumulator
        self.acc.data = self.acc.add(input.mul(2).sub(1).type(self.buftype)).clamp(0, self.buf_max.item())
        return output
    
    def UnarySign_forward_rc_sr(self, input):
        # check whether sr sum is less than half, i.e., bipolar zero.
        output = torch.lt(self.sr_cnt, self.depth_half).type(self.bstype)
        # update shiftreg
        _, self.sr_cnt.data = self.shiftreg(input)
        return output

    def forward(self, input):
        if self.sr is False:
            return self.UnarySign_forward_rc(input)
        else:
            return self.UnarySign_forward_rc_sr(input)
        