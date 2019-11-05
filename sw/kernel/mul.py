import torch
from UnarySim.sw.bitstream.gen import RNG, SourceGen, BSGen

class UnaryMul(torch.nn.Module):
    """
    this module is for unary multiplication, supporting static/non-static computation, unipolar/bipolar
    input_prob is a don't care if the multiplier is non-static, defualt value is 0.
    if the multiplier is static, then need to input the pre-scaled input_1 to port input_prob 
    """
    def __init__(self,
                 bitwidth=8,
                 mode="bipolar",
                 static=True,
                 input_prob=None):
        super(UnaryMul, self).__init__()
        
        self.bitwidth = bitwidth
        self.mode = mode
        self.static = static

        # the probability of input_1 used in static computation
        self.input_prob = input_prob
        
        # the random number generator used in computation
        self.rng = RNG(
            bitwidth=self.bitwidth,
            dim=1,
            mode="Sobol")()
        
        # currently only support static mode
        if self.static is True:
            # directly create an unchange bitstream generator for static computation
            self.source_gen = SourceGen(self.input_prob, self.bitwidth, self.mode)()
            self.bs = BSGen(self.source_gen, self.rng)
            # rng_idx is used later as a enable signal, get update every cycled
            self.rng_idx = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)
            
            # Generate two seperate bitstream generators and two enable signals for bipolar mode
            if self.mode is "bipolar":
                self.bs_inv = BSGen(self.source_gen, self.rng)
                self.rng_idx_inv = torch.nn.Parameter(torch.zeros(1).type(torch.long), requires_grad=False)

        # support of in-stream mode will be updated later
        # else:
        #     # need to count for one in bs every cycle and update probability for non-static computation    
        #     self.cnt = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        #     self.gen_prob = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        #     if self.mode is "unipolar":

        #         # this time the source_gen and bs need to be update every time by the updated prob
        #         self.source_gen = SourceGen(self.gen_prob,self.bitwidth,"unipolar")()
        #         self.bs = BSGen(self.source_gen,self.rng)
        #         self.rng_idx = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        #     else:
        #         self.source_gen = SourceGen(self.gen_prob,self.bitwidth,"bipolar")()
        #         self.bs_0 = BSGen(self.source_gen,self.rng)
        #         self.bs_1 = BSGen(self.source_gen,self.rng)
        #         self.rng_idx_0 = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        #         self.rng_idx_1 = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def UnaryMul_forward(self, input_0, input_1=None):
        # currently only support static mode
        if self.static is True:
            path_0 = input_0 & self.bs(self.rng_idx)
            # update rng index according to current input0. The update simulates enable signal of bs gen
            self.rng_idx.data = self.rng_idx.add(input_0.type(torch.long))
            
            if self.mode is "unipolar":
                return path_0
            elif self.mode is "bipolar":
                path_1 = (1 - input_0) & (1 - self.bs_inv(self.rng_idx_inv))
                # update two rng_index
                self.rng_idx_inv.data = self.rng_idx_inv.add(1 - input_0.type(torch.long))
                return path_0 | path_1
            else:
                raise ValueError("UnaryMul mode is not implemented.")
            
        # support of in-stream mode will be updated later 
        # else:

        #     if self.mode is "unipolar":    

        #         self.cnt.data =self.cnt.add(input_1) # update the number of 1 in input_1 every cycle
        #         self.gen_prob.data = self.cnt.div(self.bitwidth) # update probability

        #         # update the source_gen and bit_stream_generator
        #         self.source_gen = SourceGen(self.gen_prob,self.bitwidth,"unipolar")
        #         self.bs = BSGen(self.source_gen,self.rng)

        #         output = input_0.mul(self.bs(self.rng_idx))
        #         # as the enable signal, if input_0 is zero, no new bit will be generated
        #         self.rng_idx.data = self.rng_idx.add(input_0.type(torch.long)) 

        #     else:
        #         # update
        #         self.cnt.data = self.cnt.add(input_1)
        #         self.gen_prob.data = self.cnt.div(self.bitwidth) # update probability
        #         self.source_gen = SourceGen(self.gen_prob,self.bitwidth,"bipolar")
        #         self.bs_0 = BSGen(self.source_gen,self.rng)
        #         self.bs_1 = BSGen(self.source_gen,self.rng)


        #         gen_0 = 1 - self.bs_0(self.rng_idx_0)
        #         self.rng_idx_0.data = self.rng_idx_0.add(1 - input_0.type(torch.long))
        #         in_0 = input_0.mul(gen_0)

        #         gen_1 = self.bs_1(self.rng_idx_1)
        #         self.rng_idx_1.data = self.rng_idx_1.add(1 - input_0.type(torch.long))
        #         in_1 = input_0.mul(gen_1)

        #         output = in_1 | in_0
            
    def forward(self, input_0, input_1=None):
        return self.UnaryMul_forward(input_0, input_1)
