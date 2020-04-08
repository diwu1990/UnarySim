import torch
from UnarySim.sw.stream.gen import RNG
from UnarySim.sw.stream.shuffle import SkewedSync, Bi2Uni, Uni2Bi
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
                 rng_dim=4, 
                 bstype=torch.float):
        super(CORDIV_kernel, self).__init__()
        self.depth = depth
        self.sr = ShiftReg(depth, bstype)
        self.rng = RNG(int(math.log2(depth)), rng_dim, rng, torch.long)()
        self.idx = torch.nn.Parameter(torch.zeros(1).type(torch.float), requires_grad=False)
        self.bstype = bstype
        self.init = torch.nn.Parameter(torch.ones(1).type(torch.bool), requires_grad=False)
        self.historic_q = torch.nn.Parameter(torch.ones(1).type(torch.bool), requires_grad=False)
        
    def forward(self, dividend, divisor):
        # generate the random number to index the shift register
        # 1) generate based on divisor value, conditional probability
        # if self.init.item() is True:
        #     self.historic_q = torch.gather(self.sr.sr, 0, self.rng[self.idx.type(torch.long)%self.depth].type(torch.long))
        #     self.init.data.fill_(False)
        # else:
        #     self.historic_q = torch.gather(self.sr.sr, 0, torch.unsqueeze(self.rng[self.idx.type(torch.long)%self.depth].type(torch.long), 0))
        # divisor_eq_1 = torch.eq(divisor, 1).type(self.bstype)
        # self.idx.data = self.idx.add(divisor_eq_1)
        
        # 2) always generating, no need to deal conditional probability
        divisor_eq_1 = torch.eq(divisor, 1).type(self.bstype)
        self.historic_q.data = self.sr.sr[self.rng[self.idx.type(torch.long)%self.depth]]
        self.idx.data = self.idx.add(1)
        
        quotient = (divisor_eq_1 * dividend + (1 - divisor_eq_1) * self.historic_q).view(dividend.size())
        
        # shift register update 
        # 1) update shift register based on whether divisor is valid
        dontcare1, dontcare2 = self.sr(quotient.type(self.bstype), mask=divisor_eq_1)
        
        # 2) always update shift register
        # dontcare1, dontcare2 = self.sr(quotient.type(self.bstype), mask=None)

        return quotient.type(self.bstype)
    
    
class UnaryDiv(torch.nn.Module):
    """
    this module is for unary div, i.e., iscbdiv.
    """
    def __init__(self, 
                 depth_abs=4, 
                 depth_kernel=4, 
                 depth_sync=2, 
                 shiftreg_abs=False, 
                 mode="bipolar", 
                 rng="Sobol", 
                 rng_dim=4, 
                 bstype=torch.float, 
                 buftype=torch.float):
        super(UnaryDiv, self).__init__()
        
        # data representation
        self.mode = mode
        self.bstype = bstype
        
        if self.mode is "bipolar":
            self.abs_dividend = UnaryAbs(depth=depth_abs, shiftreg=shiftreg_abs, bstype=bstype, buftype=buftype)
            self.abs_divisor  = UnaryAbs(depth=depth_abs, shiftreg=shiftreg_abs, bstype=bstype, buftype=buftype)
            self.bi2uni_dividend = Bi2Uni(bstype=bstype)
            self.bi2uni_divisor  = Bi2Uni(bstype=bstype)
            self.uni2bi_quotient = Uni2Bi(bstype=bstype)
            
        self.ssync = SkewedSync(depth=depth_sync, bstype=bstype, buftype=buftype)
        self.cordiv_kernel = CORDIV_kernel(depth=depth_kernel, rng=rng, rng_dim=rng_dim, bstype=bstype)
        
    def bipolar_forward(self, dividend, divisor):
        sign_dividend, abs_dividend = self.abs_dividend(dividend)
        sign_divisor, abs_divisor = self.abs_divisor(divisor)
        uni_abs_dividend = self.bi2uni_dividend(abs_dividend)
        uni_abs_divisor = self.bi2uni_divisor(abs_divisor)
        uni_abs_quotient = self.unipolar_forward(uni_abs_dividend, uni_abs_divisor)
        bi_abs_quotient = self.uni2bi_quotient(uni_abs_quotient)
        bi_quotient = sign_dividend.type(torch.int8) ^ sign_divisor.type(torch.int8) ^ bi_abs_quotient.type(torch.int8)
        return bi_quotient
    
    def unipolar_forward(self, dividend, divisor):
        dividend_sync, divisor_sync = self.ssync(dividend, divisor)
        quotient = self.cordiv_kernel(dividend_sync, divisor_sync)
        return quotient

    def forward(self, dividend, divisor):
        if self.mode is "bipolar":
            output = self.bipolar_forward(dividend, divisor)
        else:
            output = self.unipolar_forward(dividend, divisor)
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
            dd_ds = 1 - (dividend.type(torch.int8) ^ divisor.type(torch.int8))
            ds_ds = 1 - (self.divisor_d ^ divisor.type(torch.int8))
            self.divisor_d.data = divisor.type(torch.int8)
            ds_ds_out = 1 - (ds_ds ^ (1 - output))
            inc = (dd_ds & ds_ds_out).type(torch.float)
            dec = ((1 - dd_ds) & (1 - ds_ds_out)).type(torch.float)
            
            #　following implementation is not good for accuracy due to fluctuation of negative output.
            # inc = dividend.type(torch.float)
            # dec = (1 - output ^ divisor.type(torch.int8)).type(torch.float)
        
        # scnt is also the same in terms of the up/down behavior and comparison
        self.scnt.data = (inc * (self.scnt + 1) + (1 - inc) * self.scnt).view(dividend.size())
        self.scnt.data = (dec * (self.scnt - 1) + (1 - dec) * self.scnt)
        self.scnt.data = self.scnt.clamp(0, self.scnt_max.item())
        
        return output.type(self.bstype)
    