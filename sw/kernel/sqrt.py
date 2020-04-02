import torch
from UnarySim.sw.bitstream.gen import RNG
from UnarySim.sw.bitstream.shuffle import Bi2Uni, Uni2Bi
from UnarySim.sw.kernel.shiftreg import ShiftReg
from UnarySim.sw.kernel.jkff import JKFF
from UnarySim.sw.kernel.div import CORDIV_kernel, UnaryDiv
import math

class UnarySqrt(torch.nn.Module):
    """
    this module is for unary square root, including iscbdiv-based and jkdiv-based.
    """
    def __init__(self, 
                 mode="bipolar", 
                 jk_trace=True, 
                 depth=4, 
                 rng="Sobol", 
                 rng_dim=4, 
                 emit=True, 
                 depth_emit=3, 
                 bstype=torch.float):
        super(UnarySqrt, self).__init__()
        
        assert math.ceil(math.log2(depth)) == math.floor(math.log2(depth)) , "Input depth needs to be power of 2."
        assert depth_emit<=7 , "Input depth_emit needs to less than 7."
        self.mode = mode
        self.bstype = bstype
        self.jk_trace = jk_trace
        self.emit = emit
        if emit is True:
            self.trace_emit = torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)
            self.emit_acc_max = torch.nn.Parameter(torch.zeros(1).fill_(2**depth_emit-1).type(torch.int8), requires_grad=False)
            print(self.emit_acc_max.item())
            self.emit_acc = torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)
            self.unidiv_emit = UnaryDiv(depth_abs=4, 
                                        depth_kernel=depth, 
                                        depth_sync=2, 
                                        shiftreg=False, 
                                        mode="unipolar", 
                                        rng=rng, 
                                        rng_dim=rng_dim, 
                                        bstype=torch.int8, 
                                        buftype=torch.float)
            if mode is "bipolar":
                self.bi2uni_emit = Bi2Uni(bstype=torch.int8)
                self.uni2bi_emit = Uni2Bi(bstype=torch.int8)
        else:
            self.trace = torch.nn.Parameter(torch.zeros(1).type(torch.int8), requires_grad=False)
            if mode is "bipolar":
                self.bi2uni = Bi2Uni(bstype=torch.int8)
            if jk_trace is True:
                self.jkff = JKFF(bstype=torch.int8)
            else:
                self.cordiv_kernel = CORDIV_kernel(depth=depth, rng=rng, rng_dim=rng_dim, bstype=torch.int8)
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
            # use UnaryDiv
            dividend = (1 - self.dff) & output
            divisor = self.dff + dividend
            
            # use historic quotient as trace
            # trace = self.cordiv_kernel.historic_q
            # _ = self.cordiv_kernel(dividend, divisor)
            
            # use actual quotient as trace
            trace = self.cordiv_kernel(dividend, divisor)
            self.dff.data = 1 - self.dff
        return trace
    
    def unipolar_trace_emit(self, output):
        # P_trace = (1-P_out)/P_out
        # use UnaryDiv
        dividend = 1 - output
        divisor = output
        trace_emit = self.unidiv_emit(dividend, divisor)
        self.emit_acc.data = self.emit_acc.add(trace_emit)
        return trace_emit

    def forward(self, input):
        if self.emit is False:
            output = ((1 - self.trace) & input.type(torch.int8)) + self.trace
            if self.mode is "bipolar":
                self.trace.data = self.bipolar_trace(output)
            else:
                self.trace.data = self.unipolar_trace(output)
        else:
            if self.mode is "bipolar":
                in_bs = self.bi2uni_emit(input)
                
                emit_acc_gt_0 = torch.gt(self.emit_acc, 0).type(torch.int8)
                # only when emit_acc is greater than 0 and input is 0, emitting is enabled.
                emit_en = (1 - in_bs.type(torch.int8)) & emit_acc_gt_0
                out_bs = in_bs.type(torch.int8) + emit_en
                # update emit_acc based on output
                dontcare = self.unipolar_trace_emit(out_bs)
                # update emit_acc based on emit_en
                self.emit_acc.data = self.emit_acc.sub(emit_en).clamp(0, self.emit_acc_max.item())
                
                output = self.uni2bi_emit(out_bs)
            else:
                emit_acc_gt_0 = torch.gt(self.emit_acc, 0).type(torch.int8)
                # only when emit_acc is greater than 0 and input is 0, emitting is enabled.
                emit_en = (1 - input.type(torch.int8)) & emit_acc_gt_0
                output = input.type(torch.int8) + emit_en
                # update emit_acc based on output
                dontcare = self.unipolar_trace_emit(output)
                # update emit_acc based on emit_en
                self.emit_acc.data = self.emit_acc.sub(emit_en).clamp(0, self.emit_acc_max.item())
        return output.type(self.bstype)
    

class GainesSqrt(torch.nn.Module):
    """
    this module is for Gaines square root.
    """
    def __init__(self, 
                 depth=5, 
                 mode="bipolar", 
                 rng="Sobol", 
                 rng_dim=1, 
                 bstype=torch.float):
        super(GainesSqrt, self).__init__()
        
        # data representation
        self.mode = mode
        self.scnt_max = torch.nn.Parameter(torch.tensor([2**depth-1]).type(torch.float), requires_grad=False)
        self.scnt = torch.nn.Parameter(torch.tensor([2**(depth-1)]).type(torch.float), requires_grad=False)
        self.rng = RNG(depth, rng_dim, rng, torch.float)()
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
    