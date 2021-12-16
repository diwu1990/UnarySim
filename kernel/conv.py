import torch
import math
import copy
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
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1, 
        bias=True, 
        padding_mode='zeros', 
        weight_ext=None, 
        bias_ext=None, 
        hwcfg={
            "width" : 8,
            "mode" : "bipolar",
            "scale" : None,
            "depth" : 12,
            "rng" : "Sobol",
            "dimr" : 1
        },
        swcfg={
            "btype" : torch.float, 
            "rtype" : torch.float, 
            "stype" : torch.float
        }):
        super(FSUConv2d, self).__init__()

        self.hwcfg = {}
        self.hwcfg["width"] = hwcfg["width"]
        self.hwcfg["mode"] = hwcfg["mode"].lower()
        self.hwcfg["scale"] = hwcfg["scale"]
        self.hwcfg["depth"] = hwcfg["depth"]
        self.hwcfg["rng"] = hwcfg["rng"].lower()
        self.hwcfg["dimr"] = hwcfg["dimr"]

        self.swcfg = {}
        self.swcfg["btype"] = swcfg["btype"]
        self.swcfg["rtype"] = swcfg["rtype"]
        self.swcfg["stype"] = swcfg["stype"]

        if (hwcfg["mode"].lower() == "bipolar") and (hwcfg["scale"] is not None) and (hwcfg["scale"] != (math.prod(num2tuple(kernel_size)) * in_channels + bias)):
            assert self.hwcfg["rng"].lower() not in ["race", "tc", "race10", "tc10"], \
                "Error: the hw config 'rng' in " + str(self) + " class should avoid ['race', 'tc', 'race10', 'tc10'] for bipolar data with non-scaled addition."

        assert self.swcfg["btype"] == torch.float, \
            "Error: the sw config 'btype' in " + str(self) + " class requires 'torch.float'."
        assert self.swcfg["rtype"] == torch.float, \
            "Error: the sw config 'rtype' in " + str(self) + " class requires 'torch.float'."
        assert self.swcfg["stype"] == torch.float, \
            "Error: the sw config 'stype' in " + str(self) + " class requires 'torch.float'."

        self.PC = FSUConv2dPC(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            bias=bias, 
            padding_mode=padding_mode, 
            weight_ext=weight_ext, 
            bias_ext=bias_ext, 
            hwcfg=self.hwcfg,
            swcfg=self.swcfg)
        
        self.scale = hwcfg["scale"]
        if self.scale is None:
            scale_add = math.prod(num2tuple(kernel_size)) * in_channels + bias
        else:
            scale_add = self.scale
        hwcfg_acc = copy.deepcopy(self.hwcfg)
        hwcfg_acc["scale"] = scale_add
        hwcfg_acc["entry"] = math.prod(num2tuple(kernel_size)) * in_channels + bias
        # the pc result is unsqueezed before fed to the accumulator, so accumulation dim of FSUAdd is 0.
        hwcfg_acc["dima"] = 0
        self.ACC = FSUAdd(
            hwcfg_acc,
            self.swcfg)

    @autocast()
    def forward(self, input, scale=None, entry=None):
        pc = self.PC(input)
        output = self.ACC(pc.unsqueeze(0), scale, entry)
        return output


class FSUConv2dPC(torch.nn.Conv2d):
    """
    This module is the parallel counter result of FSUConv2dPC before generating the bitstreams.
    The allowed coding for input, weight and bias with guaranteed accuracy can have the following three options.
    (input, weight, bias):
    1) rate, rate, rate
    2) rate, temporal, rate
    3) temporal, rate, rate
    However, this module itself does not force the input coding. Thus, above coding constraints should be done by users.
    """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1, 
        bias=True, 
        padding_mode='zeros', 
        weight_ext=None, 
        bias_ext=None, 
        hwcfg={
            "width" : 8,
            "mode" : "bipolar",
            "rng" : "Sobol",
            "dimr" : 1
        },
        swcfg={
            "btype" : torch.float, 
            "rtype" : torch.float, 
            "stype" : torch.float
        }):
        super(FSUConv2dPC, self).__init__(in_channels, out_channels, kernel_size, 
                                        stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        
        self.hwcfg = {}
        self.hwcfg["width"] = hwcfg["width"]
        self.hwcfg["mode"] = hwcfg["mode"].lower()
        self.hwcfg["rng"] = hwcfg["rng"].lower()
        self.hwcfg["dimr"] = hwcfg["dimr"]

        self.swcfg = {}
        self.swcfg["btype"] = swcfg["btype"]
        self.swcfg["rtype"] = swcfg["rtype"]
        self.swcfg["stype"] = swcfg["stype"]
        
        self.mode = hwcfg["mode"].lower()
        assert self.mode in ["unipolar", "bipolar"], \
            "Error: the hw config 'mode' in " + str(self) + " class requires one of ['unipolar', 'bipolar']."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        assert groups == 1, \
            "Error: the 'groups' in " + str(self) + " class requires to be 1."
        assert padding_mode == 'zeros', \
            "Error: the 'padding_mode' in " + str(self) + " class requires to be 'zeros'."

        # bias indication for original linear layer
        self.has_bias = bias
        
        # RNG for weight
        hwcfg_wrng = {
            "width" : hwcfg["width"],
            "rng" : hwcfg["rng"],
            "dimr" : hwcfg["dimr"]
        }
        self.wrng = RNG(hwcfg_wrng, swcfg)()
        if hwcfg["rng"].lower() in ["race", "tc", "race10", "tc10"]:
            self.wtc = True
        else:
            self.wtc = False

        # define the linear weight and bias
        if weight_ext is not None:
            assert (weight_ext.size()[0], weight_ext.size()[1], weight_ext.size()[2], weight_ext.size()[3]) == (out_channels, in_channels, num2tuple(kernel_size)[0], num2tuple(kernel_size)[1]), \
                "Error: the hw config 'out_channels, in_channels, kernel_size' in " + str(self) + " class unmatches the binary weight shape."
            self.weight.data = BinGen(weight_ext, self.hwcfg, self.swcfg)()
        
        if bias and (bias_ext is not None):
            assert bias_ext.size()[0] == out_channels, \
                "Error: the hw config 'out_channels' in " + str(self) + " class unmatches the binary bias shape."
            self.bias.data = BinGen(bias_ext, self.hwcfg, self.swcfg)()
            # RNG for bias, same as RNG for weight
            hwcfg_brng = {
                "width" : hwcfg["width"],
                "rng" : hwcfg["rng"],
                "dimr" : hwcfg["dimr"]
            }
            self.brng = RNG(hwcfg_brng, swcfg)()

        # define the kernel linear for input bit 1
        self.wbsg_i1 = BSGen(self.weight.view(1, self.weight.size()[0], -1), self.wrng, swcfg)
        self.wrdx_i1 = torch.nn.Parameter(torch.zeros_like(self.weight, dtype=torch.long), requires_grad=False).view(1, self.weight.size()[0], -1)
        if self.has_bias is True:
            self.bbsg = BSGen(self.bias, self.brng, swcfg)
            self.brdx = torch.nn.Parameter(torch.zeros_like(self.bias, dtype=torch.long), requires_grad=False)
        
        # if bipolar, define a kernel for input bit 0, note that there is no bias required for this kernel
        if (self.mode == "bipolar") and (self.wtc is False):
            self.wbsg_i0 = BSGen(self.weight.view(1, self.weight.size()[0], -1), self.wrng, swcfg)
            self.wrdx_i0 = torch.nn.Parameter(torch.zeros_like(self.weight, dtype=torch.long), requires_grad=False).view(1, self.weight.size()[0], -1)
            
        # indicator of even/odd cycle
        self.even_cycle_flag = torch.nn.Parameter(torch.ones(1, dtype=torch.bool), requires_grad=False)
        self.padding_0 = torch.nn.ConstantPad2d(self.padding, 0)
        self.padding_1 = torch.nn.ConstantPad2d(self.padding, 1)
        self.bipolar_mode = torch.nn.Parameter(torch.tensor([self.mode == "bipolar"], dtype=torch.bool), requires_grad=False)

    def FSUConv2d_PC_wrc(self, input):
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
        wbit_i1 = self.wbsg_i1(self.wrdx_i1).type(torch.float)
        if wbit_i1.size()[0] != batch:
            wbit_i1 = torch.cat(batch*[wbit_i1], 0)
            self.wrdx_i1 = torch.cat(batch*[self.wrdx_i1], 0)
        torch.add(self.wrdx_i1, input_reshape.type(torch.long), out=self.wrdx_i1)
        
        ibit_i1 = input_reshape.type(torch.float)
        obin_i1 = torch.empty(0, device=input.device)
        torch.matmul(ibit_i1, wbit_i1.transpose(1, 2), out=obin_i1)
        obin_i1.squeeze_(1)
        
        obin_reshape_i1 = obin_i1.reshape(input.size()[0], -1, obin_i1.size()[-1])
        obin_transpose_i1 = obin_reshape_i1.transpose(1, 2)
        obin_fold_i1 = torch.nn.functional.fold(obin_transpose_i1, output_size, (1, 1))

        if self.has_bias is True:
            bbit = self.bbsg(self.brdx).type(torch.float)
            self.brdx.add_(1)
            obin_fold_i1 += bbit.view(1, -1, 1, 1).expand_as(obin_fold_i1)

        if self.mode == "unipolar":
            return obin_fold_i1
        
        if self.mode == "bipolar":
            # generate weight and bias bits for current cycle
            wbit_i0 = 1 - self.wbsg_i0(self.wrdx_i0).type(torch.float)
            if wbit_i0.size()[0] != batch:
                wbit_i0 = torch.cat(batch*[wbit_i0], 0)
                self.wrdx_i0 = torch.cat(batch*[self.wrdx_i0], 0)
            torch.add(self.wrdx_i0, 1 - input_reshape.type(torch.long), out=self.wrdx_i0)
            
            ibit_i0 = 1 - ibit_i1
            obin_i0 = torch.empty(0, device=input.device)
            torch.matmul(ibit_i0, wbit_i0.transpose(1, 2), out=obin_i0)
            obin_i0.squeeze_(1)
            
            obin_reshape_i0 = obin_i0.reshape(input.size()[0], -1, obin_i0.size()[-1])
            obin_transpose_i0 = obin_reshape_i0.transpose(1, 2)
            obin_fold_i0 = torch.nn.functional.fold(obin_transpose_i0, output_size, (1, 1))

            return obin_fold_i1 + obin_fold_i0
    
    def FSUConv2d_PC_wtc(self, input):
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
        wbit_i1 = self.wbsg_i1(self.wrdx_i1).type(torch.float)
        if wbit_i1.size()[0] != batch:
            wbit_i1 = torch.cat(batch*[wbit_i1], 0)
            self.wrdx_i1 = torch.cat(batch*[self.wrdx_i1], 0)
        torch.add(self.wrdx_i1, torch.ones_like(input_reshape).type(torch.long), out=self.wrdx_i1)
        
        ibit_i1 = input_reshape.type(torch.float)
        obin_i1 = torch.empty(0, device=input.device)
        torch.matmul(ibit_i1, wbit_i1.transpose(1, 2), out=obin_i1)
        obin_i1.squeeze_(1)
        
        obin_reshape_i1 = obin_i1.reshape(input.size()[0], -1, obin_i1.size()[-1])
        obin_transpose_i1 = obin_reshape_i1.transpose(1, 2)
        obin_fold_i1 = torch.nn.functional.fold(obin_transpose_i1, output_size, (1, 1))

        if self.has_bias is True:
            bbit = self.bbsg(self.brdx).type(torch.float)
            self.brdx.add_(1)
            obin_fold_i1 += bbit.view(1, -1, 1, 1).expand_as(obin_fold_i1)

        if self.mode == "unipolar":
            return obin_fold_i1
        
        if self.mode == "bipolar":
            # generate weight and bias bits for current cycle
            wbit_i0 = 1 - wbit_i1
            ibit_i0 = 1 - ibit_i1
            obin_i0 = torch.empty(0, device=input.device)
            torch.matmul(ibit_i0, wbit_i0.transpose(1, 2), out=obin_i0)
            obin_i0.squeeze_(1)
            
            obin_reshape_i0 = obin_i0.reshape(input.size()[0], -1, obin_i0.size()[-1])
            obin_transpose_i0 = obin_reshape_i0.transpose(1, 2)
            obin_fold_i0 = torch.nn.functional.fold(obin_transpose_i0, output_size, (1, 1))

            return obin_fold_i1 + obin_fold_i0

    @autocast()
    def forward(self, input):
        if self.wtc:
            return self.FSUConv2d_PC_wtc(input).type(self.swcfg["stype"])
        else:
            return self.FSUConv2d_PC_wrc(input).type(self.swcfg["stype"])


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
                    weight_ext=None, 
                    bias_ext=None, 
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
        if weight_ext is not None:
            self.weight.data = weight_ext
        
        if bias and (bias_ext is not None):
            self.bias.data = bias_ext
            
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
                    weight_ext=None, 
                    bias_ext=None, 
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
        if weight_ext is not None:
            self.weight.data = weight_ext
        
        if bias and (bias_ext is not None):
            self.bias.data = bias_ext
            
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

