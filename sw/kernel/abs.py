import torch
from UnarySim.sw.kernel.shiftreg import ShiftReg

class UnaryAbs(torch.nn.Module):
    """
    this module is to calculate the bipolar absolute value unary data, based on non-scaled unipolar unary addition.
    only works for rate-coded data.
    """
    def __init__(self, depth=8, shiftreg=False, bstype=torch.float, buftype=torch.float):
        super(UnaryAbs, self).__init__()
        self.depth = depth
        self.depth_half = torch.nn.Parameter(torch.zeros(1).fill_(depth/2).type(buftype), requires_grad=False)
        self.sr = shiftreg
        self.bstype = bstype
        self.buftype = buftype
        if shiftreg is True:
            assert depth <= 127, "When using shift register implementation, buffer depth should be less than 127."
            self.shiftreg = ShiftReg(depth, self.bstype)
            self.sr_cnt = torch.nn.Parameter(torch.zeros(1).type(self.bstype), requires_grad=False)
            self.init = True
        else:
            self.buf_max = torch.nn.Parameter(torch.zeros(1).fill_(2**depth - 1).type(buftype), requires_grad=False)
            self.buf_half = torch.nn.Parameter(torch.zeros(1).fill_(2**(depth - 1)).type(buftype), requires_grad=False)
            self.acc = torch.nn.Parameter(torch.zeros(1).fill_(2**(depth - 1)).type(buftype), requires_grad=False)
    
    def UnaryAbs_forward_rc(self, input):
        # check whether acc is larger than or equal to half.
        half_prob_flag = torch.ge(self.acc, self.buf_half).type(torch.int8)
        # only when input is 0 and flag is 1, output 0; otherwise 1
        input_int8 = input.type(torch.int8)
        output = (half_prob_flag & input_int8) | ((1 - half_prob_flag) & (1 - input_int8))
        # update the accumulator
        self.acc.data = self.acc.add(input.mul(2).sub(1).type(self.buftype)).clamp(0, self.buf_max.item())
        return output.type(self.bstype)
    
    def UnaryAbs_forward_rc_sr(self, input):
        # check whether sr sum is larger than or equal to half.
        if self.init is True:
            output = torch.ones_like(input).type(self.bstype)
            self.init = False
        else:
            half_prob_flag = torch.ge(self.sr_cnt, self.depth_half).type(torch.int8)
            input_int8 = input.type(torch.int8)
            output = (half_prob_flag & input_int8) | ((1 - half_prob_flag) & (1 - input_int8))
        # update shiftreg
        _, self.sr_cnt.data = self.shiftreg(input)
        return output.type(self.bstype)

    def forward(self, input):
        if self.sr is False:
            return self.UnaryAbs_forward_rc(input)
        else:
            return self.UnaryAbs_forward_rc_sr(input)
