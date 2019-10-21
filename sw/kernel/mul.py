import torch
from UnarySim.sw.bitstream.gen import RNG, SourceGen, BSGen

class UnaryAdd(torch.nn.Module):
    """
    this module is for unary addition,
    1) accumulation mode
    2) unary data mode
    3) binary data width
    """
    def __init__(self, 
                 bitwidth=8, 
                 mode="bipolar", 
                 scaled=True, 
                 static=True, 
                 st_in=None, 
                 acc_dim=0):
        super(UnaryAdd, self).__init__()
        
        # data bit width
        self.bitwidth = bitwidth
        # data representation
        self.mode = mode
        # whether it is scaled addition
        self.scaled = scaled
        # whether one input is stored statically in the counter
        self.static = static
        # define the static input
        self.st_in = SourceGen(st_in, bitwidth=self.bitwidth, mode=mode)()
        # dimension to do reduce sum
        self.acc_dim = acc_dim
        
        # upper bound for accumulation counter in non-scaled mode
        # it is the number of input
        self.acc_bound = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.acc_bound.add_(in_features)
        
        # accumulation offset
        self.offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if mode is "unipolar":
            pass
        elif mode is "bipolar":
            self.offset.add_((in_features-1)/2)
        else:
            raise ValueError("UnaryAdd mode is not implemented.")
        
        
        # random_sequence from sobol RNG
        self.rng = RNG(self.bitwidth, 1, "Sobol")()
        

        # define the conditional updated bit stream for logic 1
        self.st_in_bs = BSGen(self.st_in, self.rng)
        self.rng_in_idx = torch.nn.Parameter(torch.zeros_like(self.st_in, dtype=torch.long), requires_grad=False)
        
        self.accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if self.scaled is False:
            self.out_accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def UnaryKernel_parallel_cnt(self, input):
        # generate input bits for current cycle
        paralle_cnt = torch.sum(self.st_in_bs(self.rng_in_idx), self.acc_dim)
        self.rng_wght_idx.add_(input.type(torch.long))
        return paralle_cnt

    def forward(self, input):
        if self.scaled is True:
            self.accumulator.data = self.accumulator.add(self.UnaryKernel_parallel_cnt(input))
            self.output = torch.ge(self.accumulator, self.acc_bound).type(torch.float)
            self.accumulator.sub_(self.output * self.acc_bound)
        else:
            self.accumulator.data = self.accumulator.add(self.UnaryKernel_parallel_cnt(input))
            self.accumulator.sub_(self.offset)
            self.output = torch.gt(self.accumulator, self.out_accumulator).type(torch.float)
            self.out_accumulator.data = self.out_accumulator.add(self.output)

        return self.output.type(torch.int8)
        