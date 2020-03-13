import torch
from UnarySim.sw.bitstream.gen import RNG, SourceGen, BSGen

class UnaryMul(torch.nn.Module):
    """
    this module is for unary multiplication, supporting static/non-static computation, unipolar/bipolar
    input_prob_1 is a don't care if the multiplier is non-static, defualt value is None.
    if the multiplier is static, then need to input the pre-scaled input_1 to port input_prob_1 
    """
    def __init__(self,
                 bitwidth=8,
                 mode="bipolar",
                 static=True,
                 input_prob_1=None,
                 bstype=torch.float,
                 randtype=torch.float):
        super(UnaryMul, self).__init__()
        
        self.bitwidth = bitwidth
        self.mode = mode
        self.static = static
        self.bstype = bstype
        self.randtype = randtype
        # the probability of input_1 used in static computation
        self.input_prob_1 = input_prob_1
        
        # the random number generator used in computation
        self.rng = RNG(
            bitwidth=self.bitwidth,
            dim=1,
            mode="Sobol",
            randtype=self.randtype)()
        
        # currently only support static mode
        if self.static is True:
            # directly create an unchange bitstream generator for static computation
            self.source_gen = SourceGen(self.input_prob_1, self.bitwidth, self.mode, self.randtype)()
            self.bs = BSGen(self.source_gen, self.rng, torch.int8)
            # rng_idx is used later as an enable signal, get update every cycled
            self.rng_idx = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)
            
            # Generate two seperate bitstream generators and two enable signals for bipolar mode
            if self.mode is "bipolar":
                self.bs_inv = BSGen(self.source_gen, self.rng, torch.int8)
                self.rng_idx_inv = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)
        else:
            raise ValueError("UnaryMul in-stream mode is not implemented.")

    def UnaryMul_forward(self, input_0, input_1=None):
        # currently only support static mode
        if self.static is True:
            # for input0 is 0.
            path_0 = input_0.type(torch.int8) & self.bs(self.rng_idx)
            # conditional update for rng index when input0 is 1. The update simulates enable signal of bs gen.
            self.rng_idx.data = self.rng_idx.add(input_0.type(torch.long))
            
            if self.mode is "unipolar":
                return path_0
            elif self.mode is "bipolar":
                # for input0 is 0.
                path_1 = (1 - input_0.type(torch.int8)) & (1 - self.bs_inv(self.rng_idx_inv))
                # conditional update for rng_idx_inv
                self.rng_idx_inv.data = self.rng_idx_inv.add(1 - input_0.type(torch.long))
                return path_0 | path_1
            else:
                raise ValueError("UnaryMul mode is not implemented.")
            
    def forward(self, input_0, input_1=None):
        return self.UnaryMul_forward(input_0, input_1).type(self.bstype)

    
class GainesMul(torch.nn.Module):
    """
    this module is for Gaines stochastic multiplication, supporting unipolar/bipolar
    """
    def __init__(self,
                 mode="bipolar",
                 bstype=torch.float):
        super(GainesMul, self).__init__()
        self.mode = mode
        self.bstype = bstype

    def UnaryMul_forward(self, input_0, input_1):
        if self.mode is "unipolar":
            return input_0.type(torch.int8) & input_1.type(torch.int8)
        elif self.mode is "bipolar":
            return 1 - (input_0.type(torch.int8) ^ input_1.type(torch.int8))
        else:
            raise ValueError("UnaryMul mode is not implemented.")
            
    def forward(self, input_0, input_1):
        return self.UnaryMul_forward(input_0, input_1).type(self.bstype)
    
    