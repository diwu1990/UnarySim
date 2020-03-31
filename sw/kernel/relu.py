import torch
from UnarySim.sw.kernel.shiftreg import ShiftReg

class UnaryReLU(torch.nn.Module):
    """
    unary ReLU activation based on comparing with bipolar 0
    data is always in bipolar representation
    the input bit streams are categorized into rate-coded and temporal-coded
    """
    def __init__(self, depth=8, bitwidth=8, encode="RC", shiftreg=False, bstype=torch.float, buftype=torch.float):
        super(UnaryReLU, self).__init__()
        self.depth = depth
        self.encode = encode
        self.sr = shiftreg
        self.bstype = bstype
        self.buftype = buftype
        if shiftreg is True:
            assert depth <= 127, "When using shift register implementation, buffer depth should be less than 127."
            self.shiftreg = ShiftReg(depth, self.bstype)
            self.depth_half = torch.nn.Parameter(torch.zeros(1).fill_(depth/2).type(buftype), requires_grad=False)
            self.sr_cnt = torch.nn.Parameter(torch.zeros(1).type(self.bstype), requires_grad=False)
            self.init = True
        if encode is "RC":
            self.buf_max = torch.nn.Parameter(torch.zeros(1).fill_(2**depth - 1).type(buftype), requires_grad=False)
            self.buf_half = torch.nn.Parameter(torch.zeros(1).fill_(2**(depth - 1)).type(buftype), requires_grad=False)
            self.acc = torch.nn.Parameter(torch.zeros(1).fill_(2**(depth - 1)).type(buftype), requires_grad=False)
        elif encode is "TC":
            self.threshold = torch.nn.Parameter(torch.zeros(1).fill_(2**(bitwidth - 1)).type(buftype), requires_grad=False)
            self.acc = torch.nn.Parameter(torch.zeros(1).type(buftype), requires_grad=False)
            self.cycle = torch.nn.Parameter(torch.zeros(1).type(buftype), requires_grad=False)
        else:
            raise ValueError("UnaryReLU encode other than \"RC\", \"TC\" is illegal.")
    
    def UnaryReLU_forward_rc(self, input):
        # check whether acc is larger than or equal to half.
        half_prob_flag = torch.ge(self.acc, self.buf_half).type(torch.int8)
        # only when input is 0 and flag is 1, output 0; otherwise 1
        output = input.type(torch.int8) | (1 - half_prob_flag)
        # update the accumulator
        self.acc.data = self.acc.add(output.mul(2).sub(1).type(self.buftype)).clamp(0, self.buf_max.item())
        return output.type(self.bstype)
    
    def UnaryReLU_forward_rc_sr(self, input):
        # check whether sr sum is larger than or equal to half.
        if self.init is True:
            output = torch.ones_like(input).type(self.bstype)
            self.init = False
        else:
            output = (torch.lt(self.sr_cnt, self.depth_half).type(torch.int8) | input.type(torch.int8)).type(self.bstype)
        # update shiftreg
        _, self.sr_cnt.data = self.shiftreg(output)
        return output.type(self.bstype)
    
    def UnaryReLU_forward_tc(self, input):
        # check reach half total cycle
        self.cycle.add_(1)
        half_cycle_flag = torch.gt(self.cycle, self.threshold).type(self.buftype)
        # check whether acc is larger than or equal to threshold, when half cycle is reached
        self.acc.data = self.acc.add(input.type(self.buftype))
        half_prob_flag = torch.gt(self.acc, self.threshold).type(self.buftype)
        # if  1
        output = (1 - half_cycle_flag) * torch.ge(self.cycle, self.acc).type(self.buftype) + half_cycle_flag * half_prob_flag * input.type(self.buftype)
        # update the accumulator
        return output.type(self.bstype)

    def forward(self, input):
        if self.encode is "RC":
            if self.sr is False:
                return self.UnaryReLU_forward_rc(input)
            else:
                return self.UnaryReLU_forward_rc_sr(input)
        elif self.encode is "TC":
            return self.UnaryReLU_forward_tc(input)

