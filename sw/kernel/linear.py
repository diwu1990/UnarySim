import torch
from UnarySim.sw.component.rng import RNG
from UnarySim.sw.component.bsgen import SourceGen, BSGen

class UnaryLinear(torch.nn.modules.linear.Linear):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 acc_bound, 
                 binary_weight=torch.tensor([0]), 
                 binary_bias=torch.tensor([0]), 
                 bitwidth=8, 
                 bias=True, 
                 mode="bipolar", 
                 scaled=True):
        super(UnaryLinear, self).__init__(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features
        # upper bound for accumulation counter in non-scaled mode
        self.acc_bound = acc_bound
        self.mode = mode
        self.scaled = scaled
        # accumulation offset
        if mode is "unipolar":
            self.offset = 0
        elif mode is "bipolar":
            self.offset = (in_features-1)/2
        else:
            raise ValueError("UnaryLinear mode is not implemented.")
        # bias indication for linear layer
        self.has_bias = bias
        # data bit width
        self.bitwidth = bitwidth
        # random_sequence from sobol RNG
        self.rng = RNG(self.bitwidth, 1, "Sobol").Out()
        # define the kernel linear
        self.kernel = torch.nn.Linear(self.in_features, self.out_features, bias=self.has_bias)
        self.buf_wght = SourceGen(binary_weight, bitwidth=self.bitwidth, mode=mode).Gen()
        self.buf_wght_bs = BSGen(self.buf_wght, self.rng)
        self.rng_wght_idx = torch.zeros(self.kernel.weight.size(), dtype=torch.long)
        if self.has_bias is True:
            self.buf_bias = SourceGen(binary_bias, bitwidth=self.bitwidth, mode=mode).Gen()
            self.buf_bias_bs = BSGen(self.buf_bias, self.rng)
            self.rng_bias_idx = torch.zeros(self.kernel.bias.size(), dtype=torch.long)
        
        # define kernel_inverse with no bias required, if bipolar
        if self.mode is "bipolar":
            self.kernel_inv = torch.nn.Linear(self.in_features, self.out_features, bias=False)
            self.buf_wght_bs_inv = BSGen(self.buf_wght, self.rng)
            self.rng_wght_idx_inv = torch.zeros(self.kernel_inv.weight.size(), dtype=torch.long)

        self.in_accumulator = torch.zeros(out_features)
        if self.scaled is False:
            self.out_accumulator = torch.zeros(out_features)
        self.output = torch.zeros(out_features)

    def UnaryKernel_accumulation(self, input):
        # generate weight bits for current cycle
        self.kernel.weight.data = self.buf_wght_bs.Gen(self.rng_wght_idx).type(torch.float)
        self.rng_wght_idx.add_(input.type(torch.long))
        if self.has_bias is True:
            self.kernel.bias.data = self.buf_bias_bs.Gen(self.rng_bias_idx).type(torch.float)
            self.rng_bias_idx.add_(1)
            
        if self.mode is "unipolar":
            return self.kernel(input)
        
        if self.mode is "bipolar":
            self.kernel_inv.weight.data = 1 - self.buf_wght_bs_inv.Gen(self.rng_wght_idx_inv).type(torch.float)
            self.rng_wght_idx_inv.add_(1 - input.type(torch.long))
            return self.kernel(input) + self.kernel_inv(1 - input)

    def forward(self, input):
        if self.scaled is True:
            self.in_accumulator = self.in_accumulator + self.UnaryKernel_accumulation(input)
            self.in_accumulator = self.in_accumulator - self.offset
            self.output = torch.gt(self.in_accumulator, self.out_accumulator).type(torch.float)
            return self.output
        else:
            self.in_accumulator = self.in_accumulator + self.UnaryKernel_accumulation(input)
            self.in_accumulator = self.in_accumulator - self.offset
            self.output = torch.gt(self.in_accumulator, self.out_accumulator).type(torch.float)
            self.out_accumulator = self.out_accumulator + self.output
            return self.output

