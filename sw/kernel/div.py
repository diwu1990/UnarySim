import torch
from UnarySim.sw.bitstream.gen import RNG
from UnarySim.sw.bitstream.shuffle import SkewedSync
from UnarySim.sw.kernel.shiftreg import ShiftReg
from UnarySim.sw.kernel.abs import UnaryAbs
import math

class CORDIV_kernel(torch.nn.Module):
    """
    the kernel of the correlated divivison
    this kernel is for unipolar only
    """
    def __init__(self, 
                 depth=4, 
                 rng="Sobol", 
                 rng_dim=1, 
                 bstype=torch.float):
        super(CORDIV_kernel, self).__init__()
        self.depth = depth
        self.sr = ShiftReg(depth, bstype)
        self.rng = RNG(int(math.log2(depth)), rng_dim, rng, torch.float)()
        self.idx = torch.nn.Parameter(torch.zeros(1).type(torch.float), requires_grad=False)
        self.bstype = bstype
        self.init = torch.nn.Parameter(torch.ones(1).type(torch.bool), requires_grad=False)
        
    def forward(self, dividend, divisor):
        # generate the random number to index the shift register
        # 1) generate based on divisor value, conditional probability
        # if self.init.item() is True:
        #     historic_q = torch.gather(self.sr.sr, 0, self.rng[self.idx.type(torch.long)%self.depth].type(torch.long))
        #     self.init.data.fill_(False)
        # else:
        #     historic_q = torch.gather(self.sr.sr, 0, torch.unsqueeze(self.rng[self.idx.type(torch.long)%self.depth].type(torch.long), 0))
        # divisor_eq_1 = torch.eq(divisor, 1).type(self.bstype)
        # self.idx.data = self.idx.add(divisor_eq_1)
        
        # 2) always generating, no need to deal conditional probability
        divisor_eq_1 = torch.eq(divisor, 1).type(self.bstype)
        historic_q = self.sr.sr[self.rng[self.idx.type(torch.long)%self.depth].type(torch.long)]
        self.idx.data = self.idx.add(1)

        quotient = (divisor_eq_1 * dividend + (1 - divisor_eq_1) * historic_q).view(dividend.size())
        
        # shift register update 
        # 1) update shift register based on whether divisor is valid
        # dontcare1, dontcare2 = self.sr(quotient.type(self.bstype), mask=divisor_eq_1)
        
        # 2) always update shift register
        dontcare1, dontcare2 = self.sr(quotient.type(self.bstype), mask=None)

        return quotient.type(self.bstype)
    
    
class UnaryDiv(torch.nn.Module):
    """
    this module is for unary div, i.e., iscbdiv.
    """
    def __init__(self, 
                 buf_dep=4, 
                 sync_dep=2, 
                 shiftreg=False, 
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
        
        if self.mode is "bipolar":
            self.abs = UnaryAbs(depth=buf_dep, 
                                bitwidth=bitwidth, 
                                shiftreg=shiftreg, 
                                bstype=bstype, 
                                buftype=buftype)
            
        self.ssync = SkewedSync(depth=sync_dep, 
                                bstype=bstype, 
                                buftype=buftype)
        
    def bipolar_forward(self, divisor, dividend):
        pass
    
    def unipolar_forward(self, divisor, dividend):
        pass

    def forward(self, divisor, dividend):
        if self.mode is "bipolar":
            output = self.bipolar_forward(divisor, dividend)
        else:
            output = self.unipolar_forward(divisor, dividend)
        return output.type(self.bstype)
    

class GainesDiv(torch.nn.Module):
    """
    this module is for Gaines division.
    """
    def __init__(self, 
                 depth=5, 
                 mode="bipolar", 
                 rng="Sobol", 
                 rng_dim=1, 
                 bstype=torch.float):
        super(GainesDiv, self).__init__()
        
        # data representation
        self.mode = mode
        self.scnt_max = torch.nn.Parameter(torch.tensor([2**depth-1]).type(torch.float), requires_grad=False)
        self.scnt = torch.nn.Parameter(torch.tensor([2**(depth-1)]).type(torch.float), requires_grad=False)
        self.rng = RNG(depth, rng_dim, rng, torch.float)()
        self.rng_idx = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)
        self.divisor_d = torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)
        self.bstype = bstype
        
    def forward(self, dividend, divisor):
        # output is the same for both bipolar and unipolar
        output = torch.gt(self.scnt, self.rng[self.rng_idx%self.rng.numel()]).type(torch.int8)
        self.rng_idx.data = self.rng_idx + 1
        output = output + torch.zeros_like(dividend, dtype=torch.int8)
        
        if self.mode is "unipolar":
            inc = dividend.type(torch.float)
            dec = (output & divisor.type(torch.int8)).type(torch.float)
        else:
            #　dd_ds = 1 - (dividend.type(torch.int8) ^ divisor.type(torch.int8))
            #　ds_ds = 1 - (self.divisor_d ^ divisor.type(torch.int8))
            #　self.divisor_d.data = divisor.type(torch.int8)
            #　ds_ds_out = 1 - (ds_ds ^ (1 - output))
            #　inc = (dd_ds & ds_ds_out).type(torch.float)
            #　dec = ((1 - dd_ds) & (1 - ds_ds_out)).type(torch.float)
            
            #　following implementation is not good for accuracy due to fluctuation of negative output.
            inc = dividend.type(torch.float)
            dec = (1 - output ^ divisor.type(torch.int8)).type(torch.float)
        
        # scnt is also the same in terms of the up/down behavior and comparison
        self.scnt.data = (inc * (self.scnt + 1) + (1 - inc) * self.scnt).view(dividend.size())
        self.scnt.data = (dec * (self.scnt - 1) + (1 - dec) * self.scnt)
        self.scnt.data = self.scnt.clamp(0, self.scnt_max.item())
        
        return output.type(self.bstype)
    