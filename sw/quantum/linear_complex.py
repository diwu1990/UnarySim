import torch
import math
from UnarySim.sw.stream.gen import RNG, RNGMulti, SourceGen, BSGen, BSGenMulti

class UnaryLinearComplex(torch.nn.Module):
    """
    this module is the matrix multiplication for unary complex data, which is always in bipolar format.
    1. input feature and weight are both complex values
    2. there is no bias
    3. accumulation always applies non-scaled addition
    """
    def __init__(self, 
                 in_features, 
                 out_features, 
                 weight_r=None, 
                 weight_i=None, 
                 bitwidth=8, 
                 scaled=False,
                 stype=torch.float,
                 buftype=torch.float,
                 randtype=torch.float):
        super(UnaryLinearComplex, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaled = scaled
        self.stype = stype
        
        # upper bound for accumulation counter in scaled mode
        self.acc_bound = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.acc_bound.add_(in_features*2)
        
        # accumulation offset for bipolar
        self.offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.offset.add_((in_features*2-1)/2)
        
        # data bit width
        self.bitwidth = bitwidth
        
        # random_sequence from sobol RNG, only one type of RNG is required
        self.rng = RNG(self.bitwidth, 1, "Sobol", randtype)()
        
        # define the convolution weight and bias
        self.buf_wght_r = SourceGen(weight_r, bitwidth=self.bitwidth, mode="bipolar", randtype=randtype)()
        self.buf_wght_i = SourceGen(weight_i, bitwidth=self.bitwidth, mode="bipolar", randtype=randtype)()

        # define the kernel linear for different parts
        # 1. real feature and real weight
        self.kernel_1_fr_wr       = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        self.buf_wght_bs_1_fr_wr  = BSGen(self.buf_wght_r, self.rng, stype=stype)
        self.rng_wght_idx_1_fr_wr = torch.nn.Parameter(torch.zeros_like(self.kernel_1_fr_wr.weight, 
                                                                        dtype=torch.long), requires_grad=False)

        self.kernel_0_fr_wr       = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        self.buf_wght_bs_0_fr_wr  = BSGen(self.buf_wght_r, self.rng, stype=stype)
        self.rng_wght_idx_0_fr_wr = torch.nn.Parameter(torch.zeros_like(self.kernel_0_fr_wr.weight, 
                                                                        dtype=torch.long), requires_grad=False)
        
        # 2. real feature and image weight
        self.kernel_1_fr_wi       = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        self.buf_wght_bs_1_fr_wi  = BSGen(self.buf_wght_i, self.rng, stype=stype)
        self.rng_wght_idx_1_fr_wi = torch.nn.Parameter(torch.zeros_like(self.kernel_1_fr_wi.weight, 
                                                                        dtype=torch.long), requires_grad=False)

        self.kernel_0_fr_wi       = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        self.buf_wght_bs_0_fr_wi  = BSGen(self.buf_wght_i, self.rng, stype=stype)
        self.rng_wght_idx_0_fr_wi = torch.nn.Parameter(torch.zeros_like(self.kernel_0_fr_wi.weight, 
                                                                        dtype=torch.long), requires_grad=False)
        
        # 3. image feature and real weight
        self.kernel_1_fi_wr       = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        self.buf_wght_bs_1_fi_wr  = BSGen(self.buf_wght_r, self.rng, stype=stype)
        self.rng_wght_idx_1_fi_wr = torch.nn.Parameter(torch.zeros_like(self.kernel_1_fi_wr.weight, 
                                                                        dtype=torch.long), requires_grad=False)

        self.kernel_0_fi_wr       = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        self.buf_wght_bs_0_fi_wr  = BSGen(self.buf_wght_r, self.rng, stype=stype)
        self.rng_wght_idx_0_fi_wr = torch.nn.Parameter(torch.zeros_like(self.kernel_0_fi_wr.weight, 
                                                                        dtype=torch.long), requires_grad=False)
        
        # 4. image feature and image weight
        self.kernel_1_fi_wi       = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        self.buf_wght_bs_1_fi_wi  = BSGen(self.buf_wght_i, self.rng, stype=stype)
        self.rng_wght_idx_1_fi_wi = torch.nn.Parameter(torch.zeros_like(self.kernel_1_fi_wi.weight, 
                                                                        dtype=torch.long), requires_grad=False)

        self.kernel_0_fi_wi       = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        self.buf_wght_bs_0_fi_wi  = BSGen(self.buf_wght_i, self.rng, stype=stype)
        self.rng_wght_idx_0_fi_wi = torch.nn.Parameter(torch.zeros_like(self.kernel_0_fi_wi.weight, 
                                                                        dtype=torch.long), requires_grad=False)
        
        # define the accumulator for real and image parts of output
        # real
        self.accumulator_r = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.out_accumulator_r = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        # image
        self.accumulator_i = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.out_accumulator_i = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def UnaryKernel_accumulation_fr_wr(self, input_fr_wr):
        # generate weight bits for current cycle
        self.kernel_1_fr_wr.weight.data =     self.buf_wght_bs_1_fr_wr(self.rng_wght_idx_1_fr_wr).type(torch.float)
        self.rng_wght_idx_1_fr_wr.add_(    input_fr_wr.type(torch.long))
        kernel_out_1_fr_wr = self.kernel_1_fr_wr(    input_fr_wr.type(torch.float))

        self.kernel_0_fr_wr.weight.data = 1 - self.buf_wght_bs_0_fr_wr(self.rng_wght_idx_0_fr_wr).type(torch.float)
        self.rng_wght_idx_0_fr_wr.add_(1 - input_fr_wr.type(torch.long))
        kernel_out_0_fr_wr = self.kernel_0_fr_wr(1 - input_fr_wr.type(torch.float))
        return kernel_out_1_fr_wr + kernel_out_0_fr_wr
        
    def UnaryKernel_accumulation_fr_wi(self, input_fr_wi):
        # generate weight bits for current cycle
        self.kernel_1_fr_wi.weight.data =     self.buf_wght_bs_1_fr_wi(self.rng_wght_idx_1_fr_wi).type(torch.float)
        self.rng_wght_idx_1_fr_wi.add_(    input_fr_wi.type(torch.long))
        kernel_out_1_fr_wi = self.kernel_1_fr_wi(    input_fr_wi.type(torch.float))

        self.kernel_0_fr_wi.weight.data = 1 - self.buf_wght_bs_0_fr_wi(self.rng_wght_idx_0_fr_wi).type(torch.float)
        self.rng_wght_idx_0_fr_wi.add_(1 - input_fr_wi.type(torch.long))
        kernel_out_0_fr_wi = self.kernel_0_fr_wi(1 - input_fr_wi.type(torch.float))
        return kernel_out_1_fr_wi + kernel_out_0_fr_wi
        
    def UnaryKernel_accumulation_fi_wr(self, input_fi_wr):
        # generate weight bits for current cycle
        self.kernel_1_fi_wr.weight.data =     self.buf_wght_bs_1_fi_wr(self.rng_wght_idx_1_fi_wr).type(torch.float)
        self.rng_wght_idx_1_fi_wr.add_(    input_fi_wr.type(torch.long))
        kernel_out_1_fi_wr = self.kernel_1_fi_wr(    input_fi_wr.type(torch.float))

        self.kernel_0_fi_wr.weight.data = 1 - self.buf_wght_bs_0_fi_wr(self.rng_wght_idx_0_fi_wr).type(torch.float)
        self.rng_wght_idx_0_fi_wr.add_(1 - input_fi_wr.type(torch.long))
        kernel_out_0_fi_wr = self.kernel_0_fi_wr(1 - input_fi_wr.type(torch.float))
        return kernel_out_1_fi_wr + kernel_out_0_fi_wr
        
    def UnaryKernel_accumulation_fi_wi(self, input_fi_wi):
        # generate weight bits for current cycle
        self.kernel_1_fi_wi.weight.data =     self.buf_wght_bs_1_fi_wi(self.rng_wght_idx_1_fi_wi).type(torch.float)
        self.rng_wght_idx_1_fi_wi.add_(    input_fi_wi.type(torch.long))
        kernel_out_1_fi_wi = self.kernel_1_fi_wi(    input_fi_wi.type(torch.float))

        self.kernel_0_fi_wi.weight.data = 1 - self.buf_wght_bs_0_fi_wi(self.rng_wght_idx_0_fi_wi).type(torch.float)
        self.rng_wght_idx_0_fi_wi.add_(1 - input_fi_wi.type(torch.long))
        kernel_out_0_fi_wi = self.kernel_0_fi_wi(1 - input_fi_wi.type(torch.float))
        return kernel_out_1_fi_wi + kernel_out_0_fi_wi

    def forward(self, input_r, input_i):
        kernel_out_fr_wr = self.UnaryKernel_accumulation_fr_wr(input_r)
        kernel_out_fr_wi = self.UnaryKernel_accumulation_fr_wi(input_r)
        kernel_out_fi_wr = self.UnaryKernel_accumulation_fi_wr(input_i)
        kernel_out_fi_wi = self.UnaryKernel_accumulation_fi_wi(1 - input_i)

        self.accumulator_r.data = self.accumulator_r.add(kernel_out_fr_wr).add(kernel_out_fi_wi)
        if self.scaled is True:
            output_r = torch.ge(self.accumulator_r, self.acc_bound).type(torch.float)
            self.accumulator_r.sub_(output_r * self.acc_bound)
        else:
            self.accumulator_r.sub_(self.offset)
            output_r = torch.gt(self.accumulator_r, self.out_accumulator_r).type(torch.float)
            self.out_accumulator_r.data = self.out_accumulator_r.add(output_r)

        self.accumulator_i.data = self.accumulator_i.add(kernel_out_fr_wi).add(kernel_out_fi_wr)
        if self.scaled is True:
            output_i = torch.ge(self.accumulator_i, self.acc_bound).type(torch.float)
            self.accumulator_i.sub_(output_i * self.acc_bound)
        else:
            self.accumulator_i.sub_(self.offset)
            output_i = torch.gt(self.accumulator_i, self.out_accumulator_i).type(torch.float)
            self.out_accumulator_i.data = self.out_accumulator_i.add(output_i)

        return output_r.type(self.stype), output_i.type(self.stype)
        

class LinearComplex(torch.nn.Module):
    """
    this module is the matrix multiplication for binary complex data.
    1. input feature and weight are both complex values
    2. there is no bias
    3. accumulation always applies non-scaled addition
    """
    def __init__(self, 
                 in_features, 
                 out_features, 
                 weight_r=None, 
                 weight_i=None):
        super(LinearComplex, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # define the kernel linear for different parts
        # 1. real feature and real weight
        self.kernel_fr_wr = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        self.kernel_fr_wr.weight.data = weight_r.clone().detach()

        # 2. real feature and image weight
        self.kernel_fr_wi = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        self.kernel_fr_wi.weight.data = weight_i.clone().detach()
        
        # 3. image feature and real weight
        self.kernel_fi_wr = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        self.kernel_fi_wr.weight.data = weight_r.clone().detach()
        
        # 4. image feature and image weight
        self.kernel_fi_wi = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        self.kernel_fi_wi.weight.data = weight_i.clone().detach()
        
    def forward(self, input_r, input_i):
        kernel_out_fr_wr = self.kernel_fr_wr(input_r)
        kernel_out_fr_wi = self.kernel_fr_wi(input_r)
        kernel_out_fi_wr = self.kernel_fi_wr(input_i)
        kernel_out_fi_wi = self.kernel_fi_wi(input_i)
        
        out_r = kernel_out_fr_wr - kernel_out_fi_wi
        out_i = kernel_out_fr_wi + kernel_out_fi_wr

        return out_r, out_i
    
