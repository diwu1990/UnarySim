import torch
import math
from UnarySim.stream.gen import RNG, RNGMulti, SourceGen, BSGen, BSGenMulti
from UnarySim.kernel.nn_utils import conv2d_output_shape
from UnarySim.kernel.linear import HUBLinearFunction
from UnarySim.kernel.linear import FxpLinearFunction
from torch.cuda.amp import autocast

# class UnaryConv2d(torch.nn.modules.conv.Conv2d):
#     """This is bipolar mul and non-scaled addition"""
#     def __init__(self, in_channels, out_channels, kernel_size, output_shape,
#                  binary_weight=torch.tensor([0]), binary_bias=torch.tensor([0]), bitwidth=8, 
#                  stride=1, padding=0, dilation=1, 
#                  groups=1, bias=True, padding_mode='zeros'):
#         super(UnaryConv2d, self).__init__(in_channels, out_channels, kernel_size)
        
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
        
#         # data bit width
#         self.buf_wght = binary_weight.clone().detach()
#         if bias is True:
#             self.buf_bias = binary_bias.clone().detach()
#         self.bitwidth = bitwidth

#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
        
#         self.groups = groups
#         self.has_bias = bias
#         self.padding_mode = padding_mode
        
#         # random_sequence from sobol RNG
#         self.rng = torch.quasirandom.SobolEngine(1).draw(pow(2,self.bitwidth)).view(pow(2,self.bitwidth))
#         # convert to bipolar
#         self.rng.mul_(2).sub_(1)
# #         print(self.rng)

#         # define the kernel linear
#         self.kernel = torch.nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, 
#                               stride=self.stride, padding=self.padding, dilation=self.dilation, 
#                               groups=self.groups, bias=self.has_bias, padding_mode=self.padding_mode)

#         # define the RNG index tensor for weight
#         self.rng_wght_idx = torch.zeros(self.kernel.weight.size(), dtype=torch.long)
#         self.rng_wght = self.rng[self.rng_wght_idx]
#         assert (self.buf_wght.size() == self.rng_wght.size()
#                ), "Input binary weight size of 'kernel' is different from true weight."
        
#         # define the RNG index tensor for bias if available, only one is required for accumulation
#         if self.has_bias is True:
#             print("Has bias.")
#             self.rng_bias_idx = torch.zeros(self.kernel.bias.size(), dtype=torch.long)
#             self.rng_bias = self.rng[self.rng_bias_idx]
#             assert (self.buf_bias.size() == self.rng_bias.size()
#                    ), "Input binary bias size of 'kernel' is different from true bias."

#         # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#         # inverse
#         # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#         # define the kernel_inverse, no bias required
#         self.kernel_inv = torch.nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, 
#                               stride=self.stride, padding=self.padding, dilation=self.dilation, 
#                               groups=self.groups, bias=False, padding_mode=self.padding_mode)
        
#         # define the RNG index tensor for weight_inverse
#         self.rng_wght_idx_inv = torch.zeros(self.kernel_inv.weight.size(), dtype=torch.long)
#         self.rng_wght_inv = self.rng[self.rng_wght_idx_inv]
#         assert (self.buf_wght.size() == self.rng_wght_inv.size()
#                ), "Input binary weight size of 'kernel_inv' is different from true weight."
        
#         self.in_accumulator = torch.zeros(output_shape)
#         self.out_accumulator = torch.zeros(output_shape)
#         self.output = torch.zeros(output_shape)
    
#     def UnaryKernel_nonscaled_forward(self, input):
#         # generate weight bits for current cycle
#         self.rng_wght = self.rng[self.rng_wght_idx]
#         self.kernel.weight.data = torch.gt(self.buf_wght, self.rng_wght).type(torch.float)
#         print(self.rng_wght_idx.size())
#         print(input.size())
#         self.rng_wght_idx.add_(input.type(torch.long))
#         if self.has_bias is True:
#             self.rng_bias = self.rng[self.rng_bias_idx]
#             self.kernel.bias.data = torch.gt(self.buf_bias, self.rng_bias).type(torch.float)
#             self.rng_bias_idx.add_(1)

#         # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#         # inverse
#         # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#         self.rng_wght_inv = self.rng[self.rng_wght_idx_inv].type(torch.float)
#         self.kernel_inv.weight.data = torch.le(self.buf_wght, self.rng_wght_inv).type(torch.float)
#         self.rng_wght_idx_inv.add_(1).sub_(input.type(torch.long))
# #         print(self.kernel(input).size())
#         return self.kernel(input) + self.kernel_inv(1-input)
    
#     def forward(self, input):
#         self.in_accumulator.add_(self.UnaryKernel_nonscaled_forward(input))
# #         .clamp_(-self.upper_bound, self.upper_bound)
#         self.in_accumulator.sub_(self.offset)
#         self.output = torch.gt(self.in_accumulator, self.out_accumulator).type(torch.float)
# #         print("accumulator result:", self.in_accumulator, self.out_accumulator)
#         self.out_accumulator.add_(self.output)
#         return self.output



class HUBConv2d(torch.nn.Conv2d):
    """
    this module is the 2d conv layer, with binary input and binary output
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride = 1, 
                 padding = 0, 
                 dilation = 1, 
                 groups = 1, 
                 bias = True, 
                 padding_mode = 'zeros', 
                 binary_weight=None, 
                 binary_bias=None, 
                 cycle=128,
                 rounding="floor"):
        super(HUBConv2d, self).__init__(in_channels, 
                                            out_channels, 
                                            kernel_size, 
                                            stride, 
                                            padding, 
                                            dilation, 
                                            groups, 
                                            bias, 
                                            padding_mode)
        
        # weight and bias
        if binary_weight is not None:
            self.weight.data = binary_weight
        
        if bias and (binary_bias is not None):
            self.bias.data = binary_bias
            
        # mac computing cycle
        self.cycle = cycle
        
        # bitwidth of rng
        self.bitwidth = (self.cycle - 1).bit_length()
        
        # random_sequence from sobol RNG
        self.rng = RNG(self.bitwidth, 1, "Sobol")()
        
        # generate the value map for mul using current rng
        # dim 0 is input index
        # the tensor input value is the actual value produced by the rng
        self.input_map = torch.nn.Parameter(torch.empty(cycle), requires_grad=False)
        input_val_cycle = torch.empty(0)
        torch.cat(cycle*[torch.as_tensor([c for c in range(cycle)], dtype=torch.float).unsqueeze(1)], 1, out=input_val_cycle)
        input_bit_cycle = torch.empty(0)
        torch.gt(input_val_cycle, self.rng.unsqueeze(0), out=input_bit_cycle)
        self.input_map.data = torch.sum(input_bit_cycle, 1).squeeze_().type(torch.long)

        # dim 0 is input index, dim 1 is weight index
        # the tensor value is the actual weight value produced by the rng, under a specific input and weight
        self.wght_map = torch.nn.Parameter(torch.empty(cycle, cycle), requires_grad=False)
        wght_bit_cycle = torch.empty(0)
        torch.gt(input_val_cycle, self.rng.unsqueeze(0), out=wght_bit_cycle)
        for c in range(cycle):
            self.wght_map.data[c] = torch.sum(wght_bit_cycle[:, 0:self.input_map.data[c]], 1).squeeze_()
        
        # rounding mode
        self.rounding = rounding
        
        self.rshift_input = None
        self.rshift_wght = None
        self.rshift_output = None
    
    @autocast()
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        with torch.no_grad():
            input_max_int = input.abs().max().log2()
            wght_max_int = self.weight.abs().max().log2()
            if self.rounding == "round":
                input_max_int = input_max_int.round()
                wght_max_int = wght_max_int.round()
            elif self.rounding == "floor":
                input_max_int = input_max_int.floor()
                wght_max_int = wght_max_int.floor()
            elif self.rounding == "ceil":
                input_max_int = input_max_int.ceil()
                wght_max_int = wght_max_int.ceil()

            self.rshift_input = input_max_int - self.bitwidth
            self.rshift_wght = wght_max_int - self.bitwidth
            self.rshift_output = self.bitwidth - input_max_int - wght_max_int
            
            # all data are in NCHW
            output_size = conv2d_output_shape((input.size()[2], input.size()[3]), kernel_size=self.kernel_size, dilation=self.dilation, pad=self.padding, stride=self.stride)

        # See the autograd section for explanation of what happens here.
        input_im2col = torch.nn.functional.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)
        input_transpose = input_im2col.transpose(1, 2)
        input_reshape = input_transpose.reshape(-1, input_transpose.size()[-1])

        weight = self.weight.view(self.weight.size(0), -1)
        mm_out = HUBLinearFunction.apply(input_reshape, weight, None, self.rshift_input, self.rshift_wght, self.rshift_output, self.cycle, self.wght_map)
        mm_out_reshape = mm_out.reshape(input.size()[0], -1, mm_out.size()[-1])
        mm_out_transpose = mm_out_reshape.transpose(1, 2)
        output = torch.nn.functional.fold(mm_out_transpose, output_size, (1, 1))

        if self.bias is None:
            return output
        else:
            return output + self.bias.view([1, self.bias.size()[0], 1, 1])


class FxpConv2d(torch.nn.Conv2d):
    """
    this module is the 2d conv layer, with binary input and binary output
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride = 1, 
                 padding = 0, 
                 dilation = 1, 
                 groups = 1, 
                 bias = True, 
                 padding_mode = 'zeros', 
                 binary_weight=None, 
                 binary_bias=None, 
                 bitwidth=8,
                 keep_res="input", # keep the resolution of input/output
                 more_res="input", # assign more resolution to input/weight
                 rounding="floor"):
        super(FxpConv2d, self).__init__(in_channels, 
                                            out_channels, 
                                            kernel_size, 
                                            stride, 
                                            padding, 
                                            dilation, 
                                            groups, 
                                            bias, 
                                            padding_mode)

        # weight and bias
        if binary_weight is not None:
            self.weight.data = binary_weight
        
        if bias and (binary_bias is not None):
            self.bias.data = binary_bias
            
        # bitwidth of abs
        if isinstance(bitwidth, tuple):
            self.bw_input, self.bw_wght = (bitwidth[0]-1, bitwidth[1]-1)
        else:
            if keep_res == "input":
                self.bw_input, self.bw_wght = (bitwidth-1, bitwidth-1)
            elif keep_res == "output":
                if bitwidth % 2 == 0:
                    self.bw_input, self.bw_wght = (int(bitwidth/2 - 1), int(bitwidth/2 - 1))
                else:
                    if more_res == "input":
                        self.bw_input, self.bw_wght = (int((bitwidth+1)/2 - 1), int((bitwidth-1)/2 - 1))
                    elif more_res == "weight":
                        self.bw_input, self.bw_wght = (int((bitwidth-1)/2 - 1), int((bitwidth+1)/2 - 1))
                    else:
                        raise ValueError("more_res should be either 'input' or 'weight' when bitwidth is not a tuple and keep_res is 'output'.")
            else:
                raise ValueError("keep_res should be either 'input' or 'output' when bitwidth is not a tuple.")
        
        # max abs value
        self.max_abs_input = 2**self.bw_input
        self.max_abs_wght = 2**self.bw_wght
        
        # rounding mode
        self.rounding = rounding
        
        self.rshift_input = None
        self.rshift_wght = None
        self.rshift_output = None
    
    @autocast()
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        with torch.no_grad():
            if self.rshift_input is None:
                input_max_int = input.abs().max().log2()
                if self.rounding == "round":
                    input_max_int = input_max_int.round()
                elif self.rounding == "floor":
                    input_max_int = input_max_int.floor()
                elif self.rounding == "ceil":
                    input_max_int = input_max_int.ceil()
                self.rshift_input = input_max_int - self.bw_input
            
            if self.rshift_wght is None:
                wght_max_int = self.weight.abs().max().log2()
                if self.rounding == "round":
                    wght_max_int = wght_max_int.round()
                elif self.rounding == "floor":
                    wght_max_int = wght_max_int.floor()
                elif self.rounding == "ceil":
                    wght_max_int = wght_max_int.ceil()
                self.rshift_wght = wght_max_int - self.bw_wght
                
            if self.rshift_output is None:
                self.rshift_output = 0 - self.rshift_input - self.rshift_wght
            
            # all data are in NCHW
            output_size = conv2d_output_shape((input.size()[2], input.size()[3]), kernel_size=self.kernel_size, dilation=self.dilation, pad=self.padding, stride=self.stride)

        # See the autograd section for explanation of what happens here.
        input_im2col = torch.nn.functional.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)
        input_transpose = input_im2col.transpose(1, 2)
        input_reshape = input_transpose.reshape(-1, input_transpose.size()[-1])

        weight = self.weight.view(self.weight.size(0), -1)
        mm_out = FxpLinearFunction.apply(input_reshape, weight, None, self.rshift_input, self.rshift_wght, self.rshift_output, self.max_abs_input, self.max_abs_wght)
        mm_out_reshape = mm_out.reshape(input.size()[0], -1, mm_out.size()[-1])
        mm_out_transpose = mm_out_reshape.transpose(1, 2)
        output = torch.nn.functional.fold(mm_out_transpose, output_size, (1, 1))

        if self.bias is None:
            return output
        else:
            return output + self.bias.view([1, self.bias.size()[0], 1, 1])