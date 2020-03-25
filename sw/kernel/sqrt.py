import torch
from UnarySim.sw.bitstream.gen import RNG
from UnarySim.sw.bitstream.shuffle import SkewedSync
from UnarySim.sw.kernel.shiftreg import ShiftReg
import math

    
class UnarySqrt(torch.nn.Module):
    """
    this module is for unary division, including iscbdiv and jkdiv.
    """
    def __init__(self, 
                 buf_dep=4, 
                 sync_dep=2, 
                 mode="bipolar", 
                 rng="Sobol", 
                 rng_dim=1, 
                 bstype=torch.float):
        super(UnaryAdd, self).__init__()
        
        # data bit width
        self.bitwidth = bitwidth
        # data representation
        self.mode = mode
        # whether it is scaled addition
        self.scaled = scaled
        # dimension to do reduce sum
        self.acc_dim = torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)
        self.acc_dim.fill_(acc_dim)
        self.bstype = bstype
        
        # upper bound for accumulation counter in non-scaled mode
        # it is the number of inputs, e.g., the size along the acc_dim dimension
        self.acc_bound = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        # accumulation offset
        self.offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        self.accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if self.scaled is False:
            self.out_accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input):
        self.acc_bound.fill_(input.size()[self.acc_dim.item()])
        if self.mode is "bipolar":
            self.offset.fill_((self.acc_bound.item()-1)/2)
        self.accumulator.data = self.accumulator.add(torch.sum(input.type(torch.float), self.acc_dim.item()))
        
        if self.scaled is True:
            output = torch.ge(self.accumulator, self.acc_bound).type(torch.float)
            self.accumulator.sub_(output * self.acc_bound)
        else:
            self.accumulator.sub_(self.offset)
            output = torch.gt(self.accumulator, self.out_accumulator).type(torch.float)
            self.out_accumulator.data = self.out_accumulator.add(output)

        return output.type(self.bstype)
    

class GainesSqrt(torch.nn.Module):
    """
    this module is for Gaines square root.
    """
    def __init__(self, 
                 buf_dep=5, 
                 mode="bipolar", 
                 rng="Sobol", 
                 rng_dim=1, 
                 bstype=torch.float):
        super(GainesSqrt, self).__init__()
        
        # data representation
        self.mode = mode
        self.scnt_max = torch.nn.Parameter(torch.tensor([2**buf_dep-1]).type(torch.float), requires_grad=False)
        self.scnt = torch.nn.Parameter(torch.tensor([2**(buf_dep-1)]).type(torch.float), requires_grad=False)
        self.rng = RNG(buf_dep, rng_dim, rng, torch.float)()
        self.rng_idx = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)
        self.out_d = torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)
        self.bstype = bstype
        
    def forward(self, input):
        # output is the same for both bipolar and unipolar
        output = torch.gt(self.scnt, self.rng[self.rng_idx%self.rng.numel()]).type(torch.int8)
        self.rng_idx.data = self.rng_idx + 1
        output = output + torch.zeros_like(input, dtype=torch.int8)
        
        if self.mode is "unipolar":
            inc = input.type(torch.float)
            dec = (output & self.out_d).type(torch.float)
            self.out_d.data = output.type(torch.int8)
        else:
            # this is not a good implementation
            # prod = 1 - output ^ self.out_d
            # inc = (input.type(torch.int8) & prod).type(torch.float)
            # dec = ((1 - input).type(torch.int8) & (1 - prod)).type(torch.float)
            # self.out_d.data = output.type(torch.int8)
            
            inc = input.type(torch.float)
            dec = (1 - output ^ self.out_d).type(torch.float)
            self.out_d.data = output.type(torch.int8)
            
        # scnt is also the same in terms of the up/down behavior and comparison
        self.scnt.data = (inc * (self.scnt + 1) + (1 - inc) * self.scnt).view(input.size())
        self.scnt.data = (dec * (self.scnt - 1) + (1 - dec) * self.scnt)
        self.scnt.data = self.scnt.clamp(0, self.scnt_max.item())
        
        return output.type(self.bstype)
    