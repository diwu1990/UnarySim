import torch
import math
from UnarySim.stream import RNG, BinGen, BSGen
from UnarySim.kernel import conv2d_output_shape, num2tuple
from UnarySim.kernel import HUBLinearFunction
from UnarySim.kernel import FXPLinearFunction
from UnarySim.kernel import FSUAdd
from torch.cuda.amp import autocast

class FSUConv2d(torch.nn.Module):
    """
    This module is for convolution with unary input and output
    """
    def __init__(self, 
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    stride=1, 
                    padding=0, 
                    dilation=1, 
                    groups=1, 
                    bias=True, 
                    padding_mode='zeros', 
                    binary_weight=None, 
                    binary_bias=None, 
                    bitwidth=8, 
                    mode="bipolar", 
                    scaled=True, 
                    scale=None, 
                    depth=12, 
                    btype=torch.float, 
                    rtype=torch.float, 
                    stype=torch.float):
        super(FSUConv2d, self).__init__()

        self.stype = stype
        self.PC = FSUConv2dPC(in_channels, 
                                out_channels, 
                                kernel_size, 
                                stride=stride, 
                                padding=padding, 
                                dilation=dilation, 
                                groups=groups, 
                                bias=bias, 
                                padding_mode=padding_mode, 
                                binary_weight=binary_weight, 
                                binary_bias=binary_bias, 
                                bitwidth=bitwidth, 
                                mode=mode, 
                                btype=btype, 
                                rtype=rtype, 
                                stype=stype)
        if scaled is True:
            if scale is None:
                scale_add = math.prod(num2tuple(kernel_size)) * in_channels + bias
            else:
                scale_add = scale
        else:
            scale_add = 1.0
        self.ACC = FSUAdd(mode=mode, 
                                scaled=scaled, 
                                scale=scale_add, 
                                dim=0, 
                                depth=depth, 
                                entry=math.prod(num2tuple(kernel_size)) * in_channels + bias, 
                                stype=stype)

    @autocast()
    def forward(self, input, scale=None, entry=None):
        pc = self.PC(input)
        output = self.ACC(pc.unsqueeze(0), scale, entry)
        return output.type(self.stype)


class FSUConv2dPC(torch.nn.Conv2d):
    """
    This module is for convolution with unary input and output
    """
    def __init__(self, 
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    stride=1, 
                    padding=0, 
                    dilation=1, 
                    groups=1, 
                    bias=True, 
                    padding_mode='zeros', 
                    binary_weight=None, 
                    binary_bias=None, 
                    bitwidth=8, 
                    mode="bipolar", 
                    btype=torch.float, 
                    rtype=torch.float, 
                    stype=torch.float):
        super(FSUConv2dPC, self).__init__(in_channels, out_channels, kernel_size, 
                                        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.has_bias = bias
        self.mode = mode
        self.stype = stype
        self.btype = btype
        self.rtype = rtype

        assert groups==1, "Supported group number is 1."
        assert padding_mode=='zeros', "Supported padding_mode number is 'zeros'."

        self.mode = mode
        
        # bias indication for original linear layer
        self.has_bias = bias
        
        # data bit width
        self.bitwidth = bitwidth
        
        # random_sequence from sobol RNG
        self.rng = RNG(self.bitwidth, 1, "Sobol")()
        
        # define the linear weight and bias
        if binary_weight is not None:
            self.weight.data = BinGen(binary_weight, bitwidth=self.bitwidth, mode=mode, rtype=rtype)()
        
        if bias and (binary_bias is not None):
            self.bias.data = BinGen(binary_bias, bitwidth=self.bitwidth, mode=mode, rtype=rtype)()

        # define the kernel linear
        self.weight_bsg = BSGen(self.weight.view(1, self.weight.size()[0], -1), self.rng, stype=stype)
        self.weight_rng_idx = torch.nn.Parameter(torch.zeros_like(self.weight, dtype=torch.long), requires_grad=False).view(1, self.weight.size()[0], -1)
        if self.has_bias is True:
            self.bias_bsg = BSGen(self.bias, self.rng, stype=stype)
            self.bias_rng_idx = torch.nn.Parameter(torch.zeros_like(self.bias, dtype=torch.long), requires_grad=False)
        
        # if bipolar, define a kernel with inverse input, note that there is no bias required for this inverse kernel
        if self.mode == "bipolar":
            self.weight_bsg_inv = BSGen(self.weight.view(1, self.weight.size()[0], -1), self.rng, stype=stype)
            self.weight_rng_idx_inv = torch.nn.Parameter(torch.zeros_like(self.weight, dtype=torch.long), requires_grad=False).view(1, self.weight.size()[0], -1)

        # indicator of even/odd cycle
        self.even_cycle_flag = torch.nn.Parameter(torch.ones(1, dtype=torch.bool), requires_grad=False)
        self.padding_0 = torch.nn.ConstantPad2d(self.padding, 0)
        self.padding_1 = torch.nn.ConstantPad2d(self.padding, 1)
        self.bipolar_mode = torch.nn.Parameter(torch.tensor([self.mode == "bipolar"], dtype=torch.bool), requires_grad=False)

    def FSUConv2d_PC(self, input):
        output_size = conv2d_output_shape((input.size()[2], input.size()[3]), kernel_size=self.kernel_size, dilation=self.dilation, pad=self.padding, stride=self.stride)

        if True in self.even_cycle_flag:
            input_padding = self.padding_0(input)
        else:
            input_padding = self.padding_1(input)

        # if unipolar mode, even_cycle_flag is always False to pad 0.
        self.even_cycle_flag.data = self.bipolar_mode ^ self.even_cycle_flag

        # See the autograd section for explanation of what happens here.
        input_im2col = torch.nn.functional.unfold(input_padding, self.kernel_size, self.dilation, 0, self.stride)
        input_transpose = input_im2col.transpose(1, 2)
        input_reshape = input_transpose.reshape(-1, 1, input_transpose.size()[-1])

        # first dim should always be batch
        batch = input_reshape.size()[0]

        # generate weight and bias bits for current cycle
        weight_bs = self.weight_bsg(self.weight_rng_idx).type(torch.float)
        if weight_bs.size()[0] != batch:
            weight_bs = torch.cat(batch*[weight_bs], 0)
            self.weight_rng_idx = torch.cat(batch*[self.weight_rng_idx], 0)
        torch.add(self.weight_rng_idx, input_reshape.type(torch.long), out=self.weight_rng_idx)
        
        kernel_out = torch.empty(0, device=input.device)
        torch.matmul(input_reshape.type(torch.float), weight_bs.transpose(1, 2), out=kernel_out)
        kernel_out.squeeze_(1)
        
        kernel_out_reshape = kernel_out.reshape(input.size()[0], -1, kernel_out.size()[-1])
        kernel_out_transpose = kernel_out_reshape.transpose(1, 2)
        kernel_out_fold = torch.nn.functional.fold(kernel_out_transpose, output_size, (1, 1))

        if self.has_bias is True:
            bias_bs = self.bias_bsg(self.bias_rng_idx).type(torch.float)
            self.bias_rng_idx.add_(1)
            kernel_out_fold += bias_bs.view(1, -1, 1, 1).expand_as(kernel_out_fold)

        if self.mode == "unipolar":
            return kernel_out_fold
        
        if self.mode == "bipolar":
            # generate weight and bias bits for current cycle
            weight_bs_inv = 1 - self.weight_bsg_inv(self.weight_rng_idx_inv).type(torch.float)
            if weight_bs_inv.size()[0] != batch:
                weight_bs_inv = torch.cat(batch*[weight_bs_inv], 0)
                self.weight_rng_idx_inv = torch.cat(batch*[self.weight_rng_idx_inv], 0)
            torch.add(self.weight_rng_idx_inv, 1 - input_reshape.type(torch.long), out=self.weight_rng_idx_inv)
            
            kernel_out_inv = torch.empty(0, device=input.device)
            torch.matmul(1 - input_reshape.type(torch.float), weight_bs_inv.transpose(1, 2), out=kernel_out_inv)
            kernel_out_inv.squeeze_(1)
            
            kernel_out_reshape_inv = kernel_out_inv.reshape(input.size()[0], -1, kernel_out_inv.size()[-1])
            kernel_out_transpose_inv = kernel_out_reshape_inv.transpose(1, 2)
            kernel_out_fold_inv = torch.nn.functional.fold(kernel_out_transpose_inv, output_size, (1, 1))

            return kernel_out_fold + kernel_out_fold_inv

    @autocast()
    def forward(self, input):
        return self.FSUConv2d_PC(input).type(self.stype)


class FSUConv2duGEMM(torch.nn.Conv2d):
    """
    This module is for convolution with unary input and output
    """
    def __init__(self, 
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    stride=1, 
                    padding=0, 
                    dilation=1, 
                    groups=1, 
                    bias=True, 
                    padding_mode='zeros', 
                    binary_weight=None, 
                    binary_bias=None, 
                    bitwidth=8, 
                    mode="bipolar", 
                    scaled=True, 
                    btype=torch.float, 
                    rtype=torch.float, 
                    stype=torch.float):
        super(FSUConv2duGEMM, self).__init__(in_channels, out_channels, kernel_size, 
                                        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.has_bias = bias
        self.mode = mode
        self.scaled = scaled
        self.stype = stype
        self.btype = btype
        self.rtype = rtype

        assert groups==1, "Supported group number is 1."
        assert padding_mode=='zeros', "Supported padding_mode number is 'zeros'."

        # upper bound for accumulation counter in scaled mode
        self.acc_bound = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.acc_bound.add_(math.prod(num2tuple(self.kernel_size)) * in_channels)
        if bias is True:
            self.acc_bound.add_(1)
            
        self.mode = mode
        self.scaled = scaled
        
        # accumulation offset
        self.offset = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if mode == "unipolar":
            pass
        elif mode == "bipolar":
            self.offset.add_((math.prod(num2tuple(self.kernel_size)) * in_channels-1)/2)
            if bias is True:
                self.offset.add_(1/2)
        else:
            raise ValueError("FSUConv2d mode is not implemented.")
        
        # bias indication for original linear layer
        self.has_bias = bias
        
        # data bit width
        self.bitwidth = bitwidth
        
        # random_sequence from sobol RNG
        self.rng = RNG(self.bitwidth, 1, "Sobol")()
        
        # define the linear weight and bias
        if binary_weight is not None:
            self.weight.data = BinGen(binary_weight, bitwidth=self.bitwidth, mode=mode, rtype=rtype)()
        
        if bias and (binary_bias is not None):
            self.bias.data = BinGen(binary_bias, bitwidth=self.bitwidth, mode=mode, rtype=rtype)()

        # define the kernel linear
        self.weight_bsg = BSGen(self.weight.view(1, self.weight.size()[0], -1), self.rng, stype=stype)
        self.weight_rng_idx = torch.nn.Parameter(torch.zeros_like(self.weight, dtype=torch.long), requires_grad=False).view(1, self.weight.size()[0], -1)
        if self.has_bias is True:
            self.bias_bsg = BSGen(self.bias, self.rng, stype=stype)
            self.bias_rng_idx = torch.nn.Parameter(torch.zeros_like(self.bias, dtype=torch.long), requires_grad=False)
        
        # if bipolar, define a kernel with inverse input, note that there is no bias required for this inverse kernel
        if self.mode == "bipolar":
            self.weight_bsg_inv = BSGen(self.weight.view(1, self.weight.size()[0], -1), self.rng, stype=stype)
            self.weight_rng_idx_inv = torch.nn.Parameter(torch.zeros_like(self.weight, dtype=torch.long), requires_grad=False).view(1, self.weight.size()[0], -1)

        self.accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        if self.scaled is False:
            self.out_accumulator = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        # indicator of even/odd cycle
        self.even_cycle_flag = torch.nn.Parameter(torch.ones(1, dtype=torch.bool), requires_grad=False)
        self.padding_0 = torch.nn.ConstantPad2d(self.padding, 0)
        self.padding_1 = torch.nn.ConstantPad2d(self.padding, 1)
        self.bipolar_mode = torch.nn.Parameter(torch.tensor([self.mode == "bipolar"], dtype=torch.bool), requires_grad=False)

    def FSUKernel_accumulation(self, input):
        output_size = conv2d_output_shape((input.size()[2], input.size()[3]), kernel_size=self.kernel_size, dilation=self.dilation, pad=self.padding, stride=self.stride)

        if True in self.even_cycle_flag:
            input_padding = self.padding_0(input)
        else:
            input_padding = self.padding_1(input)

        # if unipolar mode, even_cycle_flag is always False to pad 0.
        self.even_cycle_flag.data = self.bipolar_mode ^ self.even_cycle_flag

        # See the autograd section for explanation of what happens here.
        input_im2col = torch.nn.functional.unfold(input_padding, self.kernel_size, self.dilation, 0, self.stride)
        input_transpose = input_im2col.transpose(1, 2)
        input_reshape = input_transpose.reshape(-1, 1, input_transpose.size()[-1])

        # first dim should always be batch
        batch = input_reshape.size()[0]

        # generate weight and bias bits for current cycle
        weight_bs = self.weight_bsg(self.weight_rng_idx).type(torch.float)
        if weight_bs.size()[0] != batch:
            weight_bs = torch.cat(batch*[weight_bs], 0)
            self.weight_rng_idx = torch.cat(batch*[self.weight_rng_idx], 0)
        torch.add(self.weight_rng_idx, input_reshape.type(torch.long), out=self.weight_rng_idx)
        
        kernel_out = torch.empty(0, device=input.device)
        torch.matmul(input_reshape.type(torch.float), weight_bs.transpose(1, 2), out=kernel_out)
        kernel_out.squeeze_(1)
        
        kernel_out_reshape = kernel_out.reshape(input.size()[0], -1, kernel_out.size()[-1])
        kernel_out_transpose = kernel_out_reshape.transpose(1, 2)
        kernel_out_fold = torch.nn.functional.fold(kernel_out_transpose, output_size, (1, 1))

        if self.has_bias is True:
            bias_bs = self.bias_bsg(self.bias_rng_idx).type(torch.float)
            self.bias_rng_idx.add_(1)
            kernel_out_fold += bias_bs.view(1, -1, 1, 1).expand_as(kernel_out_fold)

        if self.mode == "unipolar":
            return kernel_out_fold
        
        if self.mode == "bipolar":
            # generate weight and bias bits for current cycle
            weight_bs_inv = 1 - self.weight_bsg_inv(self.weight_rng_idx_inv).type(torch.float)
            if weight_bs_inv.size()[0] != batch:
                weight_bs_inv = torch.cat(batch*[weight_bs_inv], 0)
                self.weight_rng_idx_inv = torch.cat(batch*[self.weight_rng_idx_inv], 0)
            torch.add(self.weight_rng_idx_inv, 1 - input_reshape.type(torch.long), out=self.weight_rng_idx_inv)
            
            kernel_out_inv = torch.empty(0, device=input.device)
            torch.matmul(1 - input_reshape.type(torch.float), weight_bs_inv.transpose(1, 2), out=kernel_out_inv)
            kernel_out_inv.squeeze_(1)
            
            kernel_out_reshape_inv = kernel_out_inv.reshape(input.size()[0], -1, kernel_out_inv.size()[-1])
            kernel_out_transpose_inv = kernel_out_reshape_inv.transpose(1, 2)
            kernel_out_fold_inv = torch.nn.functional.fold(kernel_out_transpose_inv, output_size, (1, 1))

            return kernel_out_fold + kernel_out_fold_inv

    @autocast()
    def forward(self, input):
        kernel_out_total = self.FSUKernel_accumulation(input)
        self.accumulator.data = self.accumulator.add(kernel_out_total)
        if self.scaled is True:
            output = torch.ge(self.accumulator, self.acc_bound).type(torch.float)
            self.accumulator.sub_(output * self.acc_bound)
        else:
            self.accumulator.sub_(self.offset)
            output = torch.gt(self.accumulator, self.out_accumulator).type(torch.float)
            self.out_accumulator.data = self.out_accumulator.add(output)

        return output.type(self.stype)


class HUBConv2d(torch.nn.Conv2d):
    """
    This module is the 2d conv layer, with binary input and binary output
    This cycle is the mac cycle using unipolar umul, i.e., half the bipolar umul. 
    As such, cycle = 2 ^ (bitwidth - 1).
    """
    def __init__(self, 
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    stride=1, 
                    padding=0, 
                    dilation=1, 
                    groups=1, 
                    bias=True, 
                    padding_mode='zeros', 
                    binary_weight=None, 
                    binary_bias=None, 
                    rng="Sobol", 
                    cycle=128,
                    rounding="round"):
        super(HUBConv2d, self).__init__(in_channels, 
                                            out_channels, 
                                            kernel_size, 
                                            stride, 
                                            padding, 
                                            dilation, 
                                            groups, 
                                            bias, 
                                            padding_mode)
        
        assert groups==1, "Supported group number is 1."
        assert padding_mode=='zeros', "Supported padding_mode number is 'zeros'."

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
        self.irng = RNG(self.bitwidth, 1, rng)()
        self.wrng = RNG(self.bitwidth, 1, "Sobol")()
        
        # generate the value map for mul using current rng
        # dim 0 is input index
        # the tensor input value is the actual value produced by the rng
        self.input_map = torch.nn.Parameter(torch.empty(cycle), requires_grad=False)
        input_val_cycle = torch.empty(0)
        torch.cat(cycle*[torch.as_tensor([c for c in range(cycle)], dtype=torch.float).unsqueeze(1)], 1, out=input_val_cycle)
        input_bit_cycle = torch.empty(0)
        torch.gt(input_val_cycle, self.irng.unsqueeze(0), out=input_bit_cycle)
        self.input_map.data = torch.sum(input_bit_cycle, 1).squeeze_().type(torch.long)

        # dim 0 is input index, dim 1 is weight index
        # the tensor value is the actual weight value produced by the rng, under a specific input and weight
        self.wght_map = torch.nn.Parameter(torch.empty(cycle, cycle), requires_grad=False)
        wght_bit_cycle = torch.empty(0)
        torch.gt(input_val_cycle, self.wrng.unsqueeze(0), out=wght_bit_cycle)
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

        weight = self.weight.view(self.weight.size()[0], -1)
        mm_out = HUBLinearFunction.apply(input_reshape, weight, None, self.rshift_input, self.rshift_wght, self.rshift_output, self.cycle, self.wght_map)
        mm_out_reshape = mm_out.reshape(input.size()[0], -1, mm_out.size()[-1])
        mm_out_transpose = mm_out_reshape.transpose(1, 2)
        output = torch.nn.functional.fold(mm_out_transpose, output_size, (1, 1))

        if self.bias is None:
            return output
        else:
            return output + self.bias.view([1, self.bias.size()[0], 1, 1])


class FXPConv2d(torch.nn.Conv2d):
    """
    This module is the 2d conv layer, with binary input and binary output
    """
    def __init__(self, 
                    in_channels, 
                    out_channels, 
                    kernel_size, 
                    stride=1, 
                    padding=0, 
                    dilation=1, 
                    groups=1, 
                    bias=True, 
                    padding_mode='zeros', 
                    binary_weight=None, 
                    binary_bias=None, 
                    bitwidth=8,
                    keep_res="input", # keep the resolution of input/output
                    more_res="input", # assign more resolution to input/weight
                    rounding="round"):
        super(FXPConv2d, self).__init__(in_channels, 
                                            out_channels, 
                                            kernel_size, 
                                            stride, 
                                            padding, 
                                            dilation, 
                                            groups, 
                                            bias, 
                                            padding_mode)

        assert groups==1, "Supported group number is 1."
        assert padding_mode=='zeros', "Supported padding_mode number is 'zeros'."

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

        weight = self.weight.view(self.weight.size()[0], -1)
        mm_out = FXPLinearFunction.apply(input_reshape, weight, None, self.rshift_input, self.rshift_wght, self.rshift_output, self.max_abs_input, self.max_abs_wght)
        mm_out_reshape = mm_out.reshape(input.size()[0], -1, mm_out.size()[-1])
        mm_out_transpose = mm_out_reshape.transpose(1, 2)
        output = torch.nn.functional.fold(mm_out_transpose, output_size, (1, 1))

        if self.bias is None:
            return output
        else:
            return output + self.bias.view([1, self.bias.size()[0], 1, 1])