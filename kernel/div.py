import torch
from UnarySim.stream import RNG
from UnarySim.stream import SkewedSync, Bi2Uni, Uni2Bi
from UnarySim.kernel import ShiftReg
from UnarySim.kernel import FSUSignAbs
import math

class CORDIV_kernel(torch.nn.Module):
    """
    The kernel of the correlated divivison, for unipolar only
    The dividend and divisor have to be synchronized before fed to this kernle
    """
    def __init__(
        self, 
        hwcfg={
            "entry" : 4,
            "rng" : "Sobol",
            "dimr" : 4
        },
        swcfg={
            "stype" : torch.float
        }):
        super(CORDIV_kernel, self).__init__()
        self.hwcfg = {}
        self.hwcfg["entry"] = hwcfg["entry"]
        self.hwcfg["rng"] = hwcfg["rng"]
        self.hwcfg["dimr"] = hwcfg["dimr"]

        self.swcfg = {}
        self.swcfg["stype"] = swcfg["stype"]

        self.entry = hwcfg["entry"]
        self.sr = ShiftReg(self.hwcfg, self.swcfg)
        hwcfg_rng = {
            "width" : int(math.log2(self.entry)),
            "dimr" : hwcfg["dimr"],
            "rng" : hwcfg["rng"]
        }
        swcfg_rng = {
            "rtype" : torch.long
        }
        self.rng = RNG(hwcfg_rng, swcfg_rng)()
        self.idx = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)
        self.stype = swcfg["stype"]
        self.historic_q = torch.nn.Parameter(torch.ones(1).type(self.stype), requires_grad=False)
        
    def forward(self, dividend, divisor):
        # generate the random number to index the shift register
        # always generating, no need to deal conditional probability
        divisor_eq_1 = torch.eq(divisor, 1).type(self.stype)
        self.historic_q.data = self.sr.sr[self.rng[self.idx%self.entry]]
        self.idx.data = self.idx.add(1)
        
        quotient = (divisor_eq_1 * dividend + (1 - divisor_eq_1) * self.historic_q).view(dividend.size())
        
        # shift register update based on whether divisor is valid
        dontcare1, dontcare2 = self.sr(quotient.type(self.stype), mask=divisor_eq_1)
        
        return quotient.type(self.stype)
    
    
class FSUDiv(torch.nn.Module):
    """
    This module is for fully streaming unary div, i.e., iscbdiv. Please refer to
    1) "In-Stream Stochastic Division and Square Root via Correlation"
    2) "In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing"
    """
    def __init__(
        self, 
        hwcfg={
            "depth_sa" : 3,
            "depth_ss" : 2,
            "entry_kn" : 2,
            "mode" : "bipolar",
            "rng" : "Sobol",
            "dimr" : 4
        },
        swcfg = {
            "stype" : torch.float,
            "btype" : torch.float
        }):
        super(FSUDiv, self).__init__()
        self.hwcfg = {}
        self.hwcfg["depth_sa"] = hwcfg["depth_sa"]
        self.hwcfg["depth_ss"] = hwcfg["depth_ss"]
        self.hwcfg["entry_kn"] = hwcfg["entry_kn"]
        self.hwcfg["mode"] = hwcfg["mode"].lower()
        self.hwcfg["rng"] = hwcfg["rng"]
        self.hwcfg["dimr"] = hwcfg["dimr"]

        self.swcfg = {}
        self.swcfg["btype"] = swcfg["btype"]
        self.swcfg["stype"] = swcfg["stype"]


        # data representation
        self.mode = hwcfg["mode"].lower()
        assert self.mode in ["unipolar", "bipolar"], \
            "Error: the hw config 'mode' in " + self + " class requires one of ['unipolar', 'bipolar']."
        self.stype = swcfg["stype"]
        self.btype = swcfg["btype"]
        self.depth_sa = hwcfg["depth_sa"]
        self.depth_ss = hwcfg["depth_ss"]
        self.entry_kn = hwcfg["entry_kn"]

        if self.mode == "bipolar":
            hwcfg_sa = {
                "depth" : self.depth_sa
            }
            self.abs_dividend = FSUSignAbs(hwcfg_sa, swcfg)
            self.abs_divisor  = FSUSignAbs(hwcfg_sa, swcfg)
            hwcfg_b2u = {
                "depth" : 3
            }
            self.bi2uni_dividend = Bi2Uni(hwcfg_b2u, swcfg)
            self.bi2uni_divisor  = Bi2Uni(hwcfg_b2u, swcfg)
            hwcfg_u2b = {
                "depth" : 4
            }
            self.uni2bi_quotient = Uni2Bi(hwcfg_u2b, swcfg)
        
        hwcfg_ss = {
            "depth" : self.depth_ss
        }
        self.ssync = SkewedSync(hwcfg_ss, swcfg)
        hwcfg_kn = {
            "entry" : self.entry_kn,
            "rng" : hwcfg["rng"],
            "dimr" : hwcfg["dimr"]
        }
        self.cordiv_kernel = CORDIV_kernel(hwcfg_kn, swcfg)
        
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
        if self.mode == "bipolar":
            output = self.bipolar_forward(dividend, divisor)
        else:
            output = self.unipolar_forward(dividend, divisor)
        return output.type(self.stype)
    
