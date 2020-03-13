import torch
from UnarySim.sw.bitstream.gen import RNG, SourceGen, BSGen
from UnarySim.sw.kernel.shiftreg import ShiftReg
import math

class CORDIV_kernel(torch.nn.Module):
    """
    the kernel of the correlated divivison
    this kernal is for unipolar only
    """
    def __init__(self, 
                 buf_dep=4, 
                 rng="Sobol", 
                 rng_dim=1, 
                 bstype=torch.float):
        super(CORDIV_kernel, self).__init__()
        self.buf_dep = buf_dep
        self.sr = ShiftReg(buf_dep, bstype)
        self.rng = RNG(int(math.log2(buf_dep)), rng_dim, rng, torch.float)()
        self.idx = torch.nn.Parameter(torch.zeros(1).type(torch.float), requires_grad=False)
        self.bstype = bstype
        self.init = torch.nn.Parameter(torch.ones(1).type(torch.bool), requires_grad=False)
        
    def forward(self, dividend, divisor):
        # generate the random number to index the shift register
        # 1) generate based on divisor value, conditional probability
        # if self.init.item() is True:
        #     historic_q = torch.gather(self.sr.sr, 0, self.rng[self.idx.type(torch.long)%self.buf_dep].type(torch.long))
        #     self.init.data.fill_(False)
        # else:
        #     historic_q = torch.gather(self.sr.sr, 0, torch.unsqueeze(self.rng[self.idx.type(torch.long)%self.buf_dep].type(torch.long), 0))
        # divisor_eq_1 = torch.eq(divisor, 1).type(self.bstype)
        # self.idx.data = self.idx.add(divisor_eq_1)
        
        # 2) always generating, no need to deal conditional probability
        divisor_eq_1 = torch.eq(divisor, 1).type(self.bstype)
        historic_q = self.sr.sr[self.rng[self.idx.type(torch.long)%self.buf_dep].type(torch.long)]
        self.idx.data = self.idx.add(1)

        quotient = (divisor_eq_1 * dividend + (1 - divisor_eq_1) * historic_q).view(dividend.size())
        
        # shift register update 
        # 1) update shift register based on whether divisor is valid
        # dontcare1, dontcare2 = self.sr(quotient.type(self.bstype), mask=divisor_eq_1)
        
        # 2) always update shift register
        dontcare1, dontcare2 = self.sr(quotient.type(self.bstype), mask=None)

        return quotient.type(self.bstype)
    
    
# class UnaryDiv(torch.nn.Module):
#     """
#     this module is for unary division, including iscbdiv and jkdiv.
#     """
#     def __init__(self, 
#                  buf_dep=8, 
#                  mode="bipolar", 
#                  scaled=True, 
#                  acc_dim=0,
#                  bstype=torch.float):
#         super(UnaryAdd, self).__init__()
        
#         # data bit width
#         self.bitwidth = bitwidth
#         # data representation
#         self.mode = mode
#         # whether it is scaled addition
#         self.scaled = scaled
#         # dimension to do reduce sum
#         self.acc_dim = torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)
#         self.acc_dim.fill_(acc_dim)
#         self.bstype = bstype
        
#         # upper bound for accumulation counter in non-scaled mode
#         # it is the number of inputs, e.g., the size along the acc_dim dimension
#         self.acc_bound = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
#         # accumulation offset
#         self.offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

#         self.accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
#         if self.scaled is False:
#             self.out_accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

#     def forward(self, input):
#         self.acc_bound.fill_(input.size()[self.acc_dim.item()])
#         if self.mode is "bipolar":
#             self.offset.fill_((self.acc_bound.item()-1)/2)
#         self.accumulator.data = self.accumulator.add(torch.sum(input.type(torch.float), self.acc_dim.item()))
        
#         if self.scaled is True:
#             output = torch.ge(self.accumulator, self.acc_bound).type(torch.float)
#             self.accumulator.sub_(output * self.acc_bound)
#         else:
#             self.accumulator.sub_(self.offset)
#             output = torch.gt(self.accumulator, self.out_accumulator).type(torch.float)
#             self.out_accumulator.data = self.out_accumulator.add(output)

#         return output.type(self.bstype)
    

class GainesDiv(torch.nn.Module):
    """
    this module is for Gaines division.
    """
    def __init__(self, 
                 buf_dep=5, 
                 mode="bipolar", 
                 rng="Sobol", 
                 rng_dim=1, 
                 bstype=torch.float):
        super(GainesDiv, self).__init__()
        
        # data representation
        self.mode = mode
        self.scnt_max = torch.nn.Parameter(torch.tensor([2**buf_dep-1]).type(torch.float), requires_grad=False)
        self.scnt = torch.nn.Parameter(torch.tensor([2**(buf_dep-1)]).type(torch.float), requires_grad=False)
        self.rng = RNG(buf_dep, rng_dim, rng, torch.float)()
        self.rng_idx = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)
        self.divisor_d = torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)
        self.bstype = bstype
        
    def forward(self, dividend, divisor):
        # output is the same for both bipolar and unipolar
        output = torch.gt(self.scnt, self.rng[self.rng_idx%self.rng.numel()]).type(torch.int8)
        self.rng_idx.data = self.rng_idx + 1
        output = output + torch.zeros_like(dividend, dtype=torch.int8)
        
        if self.mode is "unipolar":
            dividend_eq_1 = torch.eq(dividend, 1)
            prod = output & divisor.type(torch.int8)
            prod_eq_1 = torch.eq(prod, 1)
            inc = dividend_eq_1.type(torch.float)
            dec = prod_eq_1.type(torch.float)
        else:
            dd_ds = 1 - (dividend.type(torch.int8) ^ divisor.type(torch.int8))
            ds_ds = 1 - (self.divisor_d ^ divisor.type(torch.int8))
            self.divisor_d.data = divisor.type(torch.int8)
            ds_ds_out = 1 - (ds_ds ^ (1 - output))
            inc = (dd_ds & ds_ds_out).type(torch.float)
            dec = ((1 - dd_ds) & (1 - ds_ds_out)).type(torch.float)
        
        # scnt is also the same in terms of the up/down behavior and comparison
        self.scnt.data = (inc * (self.scnt + 1) + (1 - inc) * self.scnt).view(dividend.size())
        self.scnt.data = (dec * (self.scnt - 1) + (1 - dec) * self.scnt)
        self.scnt.data = self.scnt.clamp(0, self.scnt_max.item())
        
        return output.type(self.bstype)
    