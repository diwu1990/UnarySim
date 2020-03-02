import torch
from UnarySim.sw.kernel.shiftreg import ShiftReg

class UnaryReLU(torch.nn.Module):
    """
    unary ReLU activation based on comparing with bipolar 0
    data is always in bipolar representation
    the input bit streams are categorized into Sobol and Race like
    """
    def __init__(self, buf_dep=8, bitwidth=8, rng="Sobol", shiftreg=False, bstype=torch.float, buftype=torch.float):
        super(UnaryReLU, self).__init__()
        self.buf_dep = buf_dep
        self.buf_dep_half = torch.nn.Parameter(torch.zeros(1).fill_(buf_dep/2).type(buftype), requires_grad=False)
        self.rng = rng
        self.sr = shiftreg
        self.bstype = bstype
        self.buftype = buftype
        if shiftreg is True:
            assert buf_dep <= 127, "When using shift register implementation, buffer depth should be less than 127."
            self.shiftreg = ShiftReg(buf_dep, self.bstype)
            self.sr_cnt = torch.nn.Parameter(torch.zeros(1).type(self.bstype), requires_grad=False)
            self.init = True
        if rng is "Sobol" or rng is "LFSR":
            self.buf_max = torch.nn.Parameter(torch.zeros(1).fill_(2**buf_dep - 1).type(buftype), requires_grad=False)
            self.buf_half = torch.nn.Parameter(torch.zeros(1).fill_(2**(buf_dep - 1)).type(buftype), requires_grad=False)
            self.acc = torch.nn.Parameter(torch.zeros(1).fill_(2**(buf_dep - 1)).type(buftype), requires_grad=False)
        elif rng is "Race":
            self.threshold = torch.nn.Parameter(torch.zeros(1).fill_(2**(bitwidth - 1)).type(buftype), requires_grad=False)
            self.acc = torch.nn.Parameter(torch.zeros(1).type(buftype), requires_grad=False)
            self.cycle = torch.nn.Parameter(torch.zeros(1).type(buftype), requires_grad=False)
        else:
            raise ValueError("UnaryReLU rng other than \"Sobol\", \"LFSR\" or \"Race\" is illegal.")
    
    def UnaryReLU_forward_sobol(self, input):
        # check whether acc is larger than or equal to half.
        half_prob_flag = torch.ge(self.acc, self.buf_half).type(torch.int8)
        # only when input is 0 and flag is 1, output 0; otherwise 1
        output = input.type(torch.int8) | (1 - half_prob_flag)
        # update the accumulator
        self.acc.data = self.acc.add(output.mul(2).sub(1).type(self.buftype)).clamp(0, self.buf_max.item())
        return output.type(self.bstype)
    
    def UnaryReLU_forward_sobol_sr(self, input):
        # check whether sr sum is larger than or equal to half.
        if self.init is True:
            output = torch.ones_like(input).type(self.bstype)
            self.init = False
        else:
            output = (torch.lt(self.sr_cnt, self.buf_dep_half).type(torch.int8) | input.type(torch.int8)).type(self.bstype)
        # update shiftreg
        _, self.sr_cnt.data = self.shiftreg(output)
        return output
    
    def UnaryReLU_forward_race(self, input):
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
        if self.rng is "Sobol" or self.rng is "LFSR":
            if self.sr is False:
                return self.UnaryReLU_forward_sobol(input)
            else:
                return self.UnaryReLU_forward_sobol_sr(input)
        elif self.rng is "Race":
            return self.UnaryReLU_forward_race(input)

