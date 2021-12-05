import torch
from UnarySim.stream import RNG
from UnarySim.stream import Bi2Uni
from UnarySim.kernel import ShiftReg
from UnarySim.kernel import JKFF
from UnarySim.kernel import CORDIV_kernel
from UnarySim.kernel import FSUAdd
import math

class FSUSqrt(torch.nn.Module):
    """
    This module is for unary square root, including iscbdiv-based and jkdiv-based. Please refer to
    1) "In-Stream Stochastic Division and Square Root via Correlation"
    2) "In-Stream Correlation-Based Division and Bit-Inserting Square Root in Stochastic Computing"
    """
    def __init__(
        self, 
        hwcfg={
            "mode" : "bipolar",
            "jk_trace" : True,
            "emit" : True,
            "entry_kn" : 1,
            "entry_sr" : 2,
            "rng" : "Sobol",
            "dimr" : 4
        },
        swcfg={
            "stype" : torch.float,
            "btype" : torch.float
        }):
        super(FSUSqrt, self).__init__()
        self.hwcfg = {}
        self.hwcfg["mode"] = hwcfg["mode"].lower()
        self.hwcfg["jk_trace"] = hwcfg["jk_trace"]
        self.hwcfg["emit"] = hwcfg["emit"]
        self.hwcfg["entry_kn"] = hwcfg["entry_kn"]
        self.hwcfg["entry_sr"] = hwcfg["entry_sr"]
        self.hwcfg["rng"] = hwcfg["rng"]
        self.hwcfg["dimr"] = hwcfg["dimr"]

        self.swcfg = {}
        self.swcfg["btype"] = swcfg["btype"]
        self.swcfg["stype"] = swcfg["stype"]

        # data representation
        self.mode = hwcfg["mode"].lower()
        assert self.mode == "unipolar" or "bipolar", \
            "Error: the hw config 'mode' in " + self + " class requires one of ['unipolar', 'bipolar']."
        
        self.jk_trace = hwcfg["jk_trace"]
        self.emit = hwcfg["emit"]
        self.entry_kn = hwcfg["entry_kn"]
        self.entry_sr = hwcfg["entry_sr"]
        assert math.ceil(math.log2(self.entry_kn)) == math.floor(math.log2(self.entry_kn)) , \
            "Eroor: the hw config 'entry_kn' in " + self + " class needs to be power of 2."
        assert math.ceil(math.log2(self.entry_sr)) == math.floor(math.log2(self.entry_sr)) , \
            "Eroor: the hw config 'entry_sr' in " + self + " class needs to be power of 2."

        self.stype = swcfg["stype"]
        if self.emit is True:
            self.emit_out = torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)
            hwcfg_add={
                "mode" : "unipolar",
                "scale" : 1,
                "dima" : 0,
                "depth" : 10,
                "entry" : None,
            }
            swcfg_add={
                "btype" : swcfg["btype"],
                "stype" : torch.int8,
            }
            self.nsadd = FSUAdd(hwcfg_add, swcfg_add)
            hwcfg_sr={
                "entry" : self.entry_sr
            }
            swcfg_sr={
                "btype" : swcfg["btype"],
                "stype" : torch.int8,
            }
            self.sr = ShiftReg(hwcfg_sr, swcfg_sr)
            hwcfg_rng={
                "width" : int(math.log2(self.entry_sr)), 
                "dimr" : 1, 
                "rng" : hwcfg["rng"]
            }
            swcfg_rng={
                "rtype" : torch.long
            }
            self.rng = RNG(hwcfg_rng, swcfg_rng)()
            self.idx = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)
            if self.mode == "bipolar":
                hwcfg_b2u={
                    "depth" : 3
                }
                swcfg_b2u={
                    "btype" : swcfg["btype"],
                    "stype" : torch.int8
                }
                self.bi2uni_emit = Bi2Uni(hwcfg_b2u, swcfg_b2u)
        else:
            self.trace = torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)
            if self.mode == "bipolar":
                hwcfg_b2u={
                    "depth" : 3
                }
                swcfg_b2u={
                    "btype" : swcfg["btype"],
                    "stype" : torch.int8
                }
                self.bi2uni = Bi2Uni(hwcfg_b2u, swcfg_b2u)
            if self.jk_trace is True:
                swcfg_jkff={
                    "stype" : torch.int8
                }
                self.jkff = JKFF(swcfg_jkff)
            else:
                hwcfg_cordiv_kernel={
                    "entry" : 4,
                    "rng" : hwcfg["rng"],
                    "dimr" : 4
                }
                swcfg_cordiv_kernel={
                    "stype" : torch.int8
                }
                self.cordiv_kernel = CORDIV_kernel(hwcfg_cordiv_kernel, swcfg_cordiv_kernel)
                self.dff = torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)
        
    def bipolar_trace(self, output):
        # P_trace = (P_out*2-1)/((P_out*2-1)+1)
        out = self.bi2uni(output)
        trace = self.unipolar_trace(out)
        return trace
    
    def unipolar_trace(self, output):
        # P_trace = P_out/(P_out+1)
        if self.jk_trace is True:
            # use JKFF
            trace = self.jkff(output, torch.ones_like(output))
        else:
            # use FSUDiv
            dividend = (1 - self.dff) & output
            divisor = self.dff + dividend
            
            # use historic quotient as trace
            # trace = self.cordiv_kernel.historic_q[0]
            # _ = self.cordiv_kernel(dividend, divisor)
            
            # use actual quotient as trace
            trace = self.cordiv_kernel(dividend, divisor)
            
            self.dff.data = 1 - self.dff
        return trace
    
    def unipolar_emit(self, output):
        output_inv = 1 - output
        output_inv_scrambled, dontcare = self.sr(output_inv, index=self.idx.item()%self.entry_sr)
        emit_out = output_inv_scrambled & output
        return emit_out
    
    def bipolar_emit(self, output):
        output_inv = 1 - output
        output_inv_scrambled, dontcare = self.sr(output_inv, index=self.idx.item()%self.entry_sr)
        output_uni = self.bi2uni_emit(output)
        emit_out = output_inv_scrambled & output_uni
        return emit_out

    def forward(self, input):
        if self.emit is True:
            if list(self.emit_out.size()) != list(input.size()):
                self.emit_out.data = torch.zeros_like(input).type(torch.int8)
            in_stack = torch.stack([input.type(torch.int8), self.emit_out], dim=0)
            output = self.nsadd(in_stack)
            if self.mode == "bipolar":
                self.emit_out.data = self.bipolar_emit(output)
            else:
                self.emit_out.data = self.unipolar_emit(output)
        else:
            output = ((1 - self.trace) & input.type(torch.int8)) + self.trace
            if self.mode == "bipolar":
                self.trace.data = self.bipolar_trace(output)
            else:
                self.trace.data = self.unipolar_trace(output)
        return output.type(self.stype)
    
