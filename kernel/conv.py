import torch
import math
import copy
from UnarySim.stream import RNG, BinGen, BSGen
from UnarySim.kernel import conv2d_output_shape, num2tuple
from UnarySim.kernel import HUBLinearFunction
from UnarySim.kernel import FXPLinearFunction
from UnarySim.kernel import TLUTLinearFXPFXPFunction, TLUTLinearFXPFPFunction, TLUTLinearFPFPFunction
from UnarySim.kernel import FSUAdd, rshift_offset
from UnarySim.kernel import NCFireStep, Round
from torch.cuda.amp import autocast
import torch.jit as jit

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
    This module is the conv2d layer for binary signed data in fxp format using unary computing.
    The hardware configuration specifies 
    1) the data with in bit for input and weight/bias
    2) the rng type for input and weight/bias
    3) the quantile to quantize input and weight/bias
    4) the cycle to early terminate the run
    5) the rounding mode for both input and weight/bias
    6) whether to use sign-magnitude format for both input and weight/bias, which can reduce the number of max cycle count to run
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
            "widthi" : 8,
            "rngi" : "Sobol",
            "quantilei" : 1,
            "widthw" : 8,
            "rngw" : "Sobol",
            "quantilew" : 1,
            "cycle" : 128,
            "scale" : 1,
            "rounding" : "round",
            "signmag" : True
        }):
        super(HUBConv2d, self).__init__(in_channels, 
                                            out_channels, 
                                            kernel_size, 
                                            stride, 
                                            padding, 
                                            dilation, 
                                            groups, 
                                            bias, 
                                            padding_mode)
        self.hwcfg = {}
        self.hwcfg["widthi"] = hwcfg["widthi"]
        self.hwcfg["rngi"] = hwcfg["rngi"].lower()
        self.hwcfg["quantilei"] = hwcfg["quantilei"]
        self.hwcfg["widthw"] = hwcfg["widthw"]
        self.hwcfg["rngw"] = hwcfg["rngw"].lower()
        self.hwcfg["quantilew"] = hwcfg["quantilew"]
        self.hwcfg["rounding"] = hwcfg["rounding"].lower()
        self.hwcfg["scale"] = hwcfg["scale"]
        self.hwcfg["signmag"] = hwcfg["signmag"]
        self.hwcfg["cycle"] = min(hwcfg["cycle"], 2**(max(hwcfg["widthi"], hwcfg["widthw"]) - hwcfg["signmag"]))

        self.itc = (self.hwcfg["rngi"] in ["race", "tc", "race10", "tc10"])
        self.wtc = (self.hwcfg["rngw"] in ["race", "tc", "race10", "tc10"])

        assert not (self.itc and self.wtc), \
            "Error: the hw config 'rngi' and 'rngw' in " + str(self) + " class can't adopt temporal coding simultaneously."

        assert self.hwcfg["quantilei"] > 0 and self.hwcfg["quantilei"] <= 1, \
            "Error: the hw config 'quantilei' of " + str(self) + " class needs to be within (0, 1]."
        
        assert self.hwcfg["quantilew"] > 0 and self.hwcfg["quantilew"] <= 1, \
            "Error: the hw config 'quantilew' of " + str(self) + " class needs to be within (0, 1]."

        assert self.hwcfg["rounding"] in ["round", "ceil", "floor"], \
            "Error: the hw config 'rounding' of " + str(self) + " class requires one of ['round', 'ceil', 'floor']."

        assert self.hwcfg["signmag"] is True, \
            "Error: the hw config 'signmag' of " + str(self) + " class requires to be True, i.e., always computing on sign-magnitue data, for diverse architectures."

        # maximum possible run cycle
        self.cycle_max = 2**(max(hwcfg["widthi"], hwcfg["widthw"]) - hwcfg["signmag"])
        # actual run cycle
        self.cycle_act = min(hwcfg["cycle"], 2**(max(hwcfg["widthi"], hwcfg["widthw"]) - hwcfg["signmag"]))

        assert groups == 1, \
            "Error: the 'groups' in " + str(self) + " class requires to be 1."
        assert padding_mode == 'zeros', \
            "Error: the 'padding_mode' in " + str(self) + " class requires to be 'zeros'."

        # weight and bias
        if weight_ext is not None:
            assert (weight_ext.size()[0], weight_ext.size()[1], weight_ext.size()[2], weight_ext.size()[3]) == (out_channels, in_channels, num2tuple(kernel_size)[0], num2tuple(kernel_size)[1]), \
                "Error: the hw config 'out_channels, in_channels, kernel_size' in " + str(self) + " class unmatches the binary weight shape."
            self.weight.data = weight_ext
        
        if bias and (bias_ext is not None):
            assert bias_ext.size()[0] == out_channels, \
                "Error: the hw config 'out_channels' in " + str(self) + " class unmatches the binary bias shape."
            self.bias.data = bias_ext

        swcfg={
            "btype" : torch.float,
            "rtype" : torch.float,
            "stype" : torch.float
        }

        # random_sequence from RNG
        hwcfg_irng = {
            "width" : self.hwcfg["widthi"] - self.hwcfg["signmag"], 
            "dimr" : 1, 
            "rng" : self.hwcfg["rngi"]
        }
        self.irng = RNG(hwcfg_irng, swcfg)()
        hwcfg_wrng = {
            "width" : self.hwcfg["widthw"] - self.hwcfg["signmag"], 
            "dimr" : 1, 
            "rng" : self.hwcfg["rngw"]
        }
        self.wrng = RNG(hwcfg_wrng, swcfg)()

        if (self.itc) and (not self.wtc):
            # cbsg controller is input
            self.rngctler = self.irng
            self.rngctlee = self.wrng
        elif (not self.itc) and (self.wtc):
            # cbsg controller is weight
            self.rngctler = self.wrng
            self.rngctlee = self.irng
        elif (not self.itc) and (not self.wtc):
            # when rate coding is applied to both input and weight, always control weight with input
            # the hardware cost of doing this is similar to the opposite
            self.rngctler = self.irng
            self.rngctlee = self.wrng
        
        # generate the value map for mul using current rng
        # dim 0 is input index
        # the tensor input value is the actual value produced by the rngctler
        self.mapctler = torch.nn.Parameter(torch.empty(self.cycle_max), requires_grad=False)
        cycle_ctlerval = torch.empty(0)
        torch.cat(self.cycle_max*[torch.arange(self.cycle_max, dtype=torch.float).unsqueeze(1)], 1, out=cycle_ctlerval)
        cycle_ctlerbit = torch.empty(0)
        torch.gt(cycle_ctlerval, self.rngctler.unsqueeze(0), out=cycle_ctlerbit)
        self.mapctler.data = torch.sum(cycle_ctlerbit, 1).squeeze_().type(torch.long)

        # dim 0 is input index, dim 1 is weight index
        # the tensor value is the actual weight value produced by the rngctlee, under a specific input and weight
        self.mapctlee = torch.nn.Parameter(torch.empty(self.cycle_max, self.cycle_max), requires_grad=False)
        cycle_ctleebit = torch.empty(0)
        torch.gt(cycle_ctlerval, self.rngctlee.unsqueeze(0), out=cycle_ctleebit)
        for c in range(self.cycle_max):
            self.mapctlee.data[c] = torch.sum(cycle_ctleebit[:, 0:self.mapctler.data[c]], 1).squeeze_()
        
        self.rshift_i = None
        self.rshift_w = None
        self.rshift_o = None
    
    @autocast()
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        self.rshift_i, self.rshift_w, self.rshift_o = \
            rshift_offset(input, self.weight, self.hwcfg["widthi"] - self.hwcfg["signmag"], self.hwcfg["widthw"] - self.hwcfg["signmag"], self.hwcfg["rounding"], self.hwcfg["quantilei"], self.hwcfg["quantilew"])
        
        with torch.no_grad():
            # all data are in NCHW
            output_size = conv2d_output_shape((input.size()[2], input.size()[3]), kernel_size=self.kernel_size, dilation=self.dilation, pad=self.padding, stride=self.stride)

        input_im2col = torch.nn.functional.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)
        input_transpose = input_im2col.transpose(1, 2)
        input_reshape = input_transpose.reshape(-1, input_transpose.size()[-1])

        weight = self.weight.view(self.weight.size()[0], -1)
        mm_out = HUBLinearFunction.apply(input_reshape, weight, None, self.rshift_i, self.rshift_w, self.rshift_o, self.cycle_act, self.mapctlee)
        mm_out_reshape = mm_out.reshape(input.size()[0], -1, mm_out.size()[-1])
        mm_out_transpose = mm_out_reshape.transpose(1, 2)
        output = torch.nn.functional.fold(mm_out_transpose, output_size, (1, 1))

        if self.bias is None:
            return output / self.hwcfg["scale"]
        else:
            return (output + self.bias.view([1, self.bias.size()[0], 1, 1])) / self.hwcfg["scale"]


class FXPConv2d(torch.nn.Conv2d):
    """
    This module is the 2d conv layer, with binary input and binary output
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
            "widthi" : 8,
            "quantilei" : 1,
            "widthw" : 8,
            "quantilew" : 1,
            "rounding" : "round"
        }):
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
        self.hwcfg = {}
        self.hwcfg["widthi"] = hwcfg["widthi"]
        self.hwcfg["quantilei"] = hwcfg["quantilei"]
        self.hwcfg["widthw"] = hwcfg["widthw"]
        self.hwcfg["quantilew"] = hwcfg["quantilew"]
        self.hwcfg["rounding"] = hwcfg["rounding"].lower()
        
        # weight and bias
        if weight_ext is not None:
            assert (weight_ext.size()[0], weight_ext.size()[1], weight_ext.size()[2], weight_ext.size()[3]) == (out_channels, in_channels, num2tuple(kernel_size)[0], num2tuple(kernel_size)[1]), \
                "Error: the hw config 'out_channels, in_channels, kernel_size' in " + str(self) + " class unmatches the binary weight shape."
            self.weight.data = weight_ext
        
        if bias and (bias_ext is not None):
            assert bias_ext.size()[0] == out_channels, \
                "Error: the hw config 'out_channels' in " + str(self) + " class unmatches the binary bias shape."
            self.bias.data = bias_ext
        
        # max abs value
        self.max_abs_i = 2**self.hwcfg["widthi"]
        self.max_abs_w = 2**self.hwcfg["widthw"]
        
        self.rshift_i = None
        self.rshift_w = None
        self.rshift_o = None
    
    @autocast()
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        self.rshift_i, self.rshift_w, _ = \
            rshift_offset(input, self.weight, self.hwcfg["widthi"] - 1, self.hwcfg["widthw"] - 1, self.hwcfg["rounding"], self.hwcfg["quantilei"], self.hwcfg["quantilew"])
        self.rshift_o = 0 - self.rshift_i - self.rshift_w

        with torch.no_grad():
            # all data are in NCHW
            output_size = conv2d_output_shape((input.size()[2], input.size()[3]), kernel_size=self.kernel_size, dilation=self.dilation, pad=self.padding, stride=self.stride)

        # See the autograd section for explanation of what happens here.
        input_im2col = torch.nn.functional.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)
        input_transpose = input_im2col.transpose(1, 2)
        input_reshape = input_transpose.reshape(-1, input_transpose.size()[-1])

        weight = self.weight.view(self.weight.size()[0], -1)
        mm_out = FXPLinearFunction.apply(input_reshape, weight, None, self.rshift_i, self.rshift_w, self.rshift_o, self.max_abs_i, self.max_abs_w)
        mm_out_reshape = mm_out.reshape(input.size()[0], -1, mm_out.size()[-1])
        mm_out_transpose = mm_out_reshape.transpose(1, 2)
        output = torch.nn.functional.fold(mm_out_transpose, output_size, (1, 1))

        if self.bias is None:
            return output
        else:
            return output + self.bias.view([1, self.bias.size()[0], 1, 1])



class TLUTConv2d(torch.nn.Conv2d):
    """
    This module is the 2d conv layer using temporal look-up table (T-LUT). 
    This module always applies sign-magnitude format for temporal bitstreams.
    The hardware configuration specifies
    1) the variable to produce temporal coded bitstreams
    2) the format with in bit for input and weight/bias, only used for temporal coded fxp data
    3) the data with in bit for input and weight/bias, as well as for the temporal coded bitstreams
    4) the quantile to quantize input and weight/bias, only used for temporal coded fxp datas
    5) the cycle to early terminate the run
    6) the rounding mode for both input and weight/bias, only used for temporal coded fxp datas
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
            "temporal" : "i",
            "widtht" : 4,
            "formati" : "fxp",
            "widthi" : 8,
            "quantilei" : 1,
            "formatw" : "fxp",
            "widthw" : 8,
            "quantilew" : 1,
            "cycle" : None,
            "rounding" : "round",
            "signmag" : True
        }):
        super(TLUTConv2d, self).__init__(in_channels, 
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
        self.hwcfg = {}
        self.hwcfg["temporal"] = hwcfg["temporal"].lower()
        self.hwcfg["widtht"] = hwcfg["widtht"]
        self.hwcfg["formati"] = hwcfg["formati"].lower()
        self.hwcfg["widthi"] = hwcfg["widthi"]
        self.hwcfg["quantilei"] = hwcfg["quantilei"]
        self.hwcfg["formatw"] = hwcfg["formatw"].lower()
        self.hwcfg["widthw"] = hwcfg["widthw"]
        self.hwcfg["quantilew"] = hwcfg["quantilew"]
        self.hwcfg["cycle"] = hwcfg["cycle"]
        self.hwcfg["rounding"] = hwcfg["rounding"].lower()
        self.hwcfg["signmag"] = hwcfg["signmag"]

        assert self.hwcfg["temporal"] in ["i", "w", "input", "weight"], \
            "Error: the hw config 'temporal' in " + str(self) + " class requires one of ['i', 'input', 'w', 'weight']."

        assert self.hwcfg["formati"] in ["fxp", "bfloat16", "float16", "float32"], \
            "Error: the hw config 'formati' in " + str(self) + " class requires one of ['fxp', 'bfloat16', 'float16', 'float32']."
        
        assert self.hwcfg["formatw"] in ["fxp", "bfloat16", "float16", "float32"], \
            "Error: the hw config 'formatw' in " + str(self) + " class requires one of ['fxp', 'bfloat16', 'float16', 'float32']."

        assert self.hwcfg["quantilei"] > 0 and self.hwcfg["quantilei"] <= 1, \
            "Error: the hw config 'quantilei' of " + str(self) + " class needs to be within (0, 1]."
        
        assert self.hwcfg["quantilew"] > 0 and self.hwcfg["quantilew"] <= 1, \
            "Error: the hw config 'quantilew' of " + str(self) + " class needs to be within (0, 1]."

        assert self.hwcfg["rounding"] in ["round", "ceil", "floor"], \
            "Error: the hw config 'rounding' of " + str(self) + " class requires one of ['round', 'ceil', 'floor']."

        assert self.hwcfg["signmag"] is True, \
            "Error: the hw config 'signmag' of " + str(self) + " class requires to be True, i.e., always computing on the magnitude of the data."
        
        # weight and bias
        if weight_ext is not None:
            assert (weight_ext.size()[0], weight_ext.size()[1], weight_ext.size()[2], weight_ext.size()[3]) == (out_channels, in_channels, num2tuple(kernel_size)[0], num2tuple(kernel_size)[1]), \
                "Error: the hw config 'out_channels, in_channels, kernel_size' in " + str(self) + " class unmatches the binary weight shape."
            self.weight.data = weight_ext
        
        if bias and (bias_ext is not None):
            assert bias_ext.size()[0] == out_channels, \
                "Error: the hw config 'out_channels' in " + str(self) + " class unmatches the binary bias shape."
            self.bias.data = bias_ext
        
        if (self.hwcfg["formati"] in ["fxp"]) and (self.hwcfg["formatw"] in ["fxp"]):
            self.mode = "fxpfxp"
        elif (self.hwcfg["formati"] not in ["fxp"]) and (self.hwcfg["formatw"] not in ["fxp"]):
            self.mode = "fpfp"
        else:
            self.mode = "fxpfp"

        # maximum possible run cycle
        self.cycle_max = 2**self.hwcfg["widtht"]
        # actual run cycle
        if self.hwcfg["cycle"] is None:
            self.cycle_act = self.cycle_max
        else:
            self.cycle_act = min(self.hwcfg["cycle"], self.cycle_max)
        self.hwcfg["cycle"] = self.cycle_act

        self.widthi = self.hwcfg["widthi"] - 1
        self.widthw = self.hwcfg["widthw"] - 1

        if self.hwcfg["temporal"] in ["i", "input"]:
            if self.hwcfg["formati"] in ["fxp"]:
                self.width = self.hwcfg["widthi"] - 1
            elif self.hwcfg["formati"] in ["bfloat16"]:
                self.width = 8
            elif self.hwcfg["formati"] in ["float16"]:
                self.width = 11
            elif self.hwcfg["formati"] in ["float32"]:
                self.width = 24
        elif self.hwcfg["temporal"] in ["w", "weight"]:
            if self.hwcfg["formatw"] in ["fxp"]:
                self.width = self.hwcfg["widthw"] - 1
            elif self.hwcfg["formatw"] in ["bfloat16"]:
                self.width = 8
            elif self.hwcfg["formatw"] in ["float16"]:
                self.width = 11
            elif self.hwcfg["formatw"] in ["float32"]:
                self.width = 24
        
        self.degree = int(math.ceil(self.width / self.hwcfg["widtht"]))
        self.delta = int(self.degree * self.hwcfg["widtht"] - self.width)
    
    def forward(self, input):
        with torch.no_grad():
            # all data are in NCHW
            output_size = conv2d_output_shape((input.size()[2], input.size()[3]), kernel_size=self.kernel_size, dilation=self.dilation, pad=self.padding, stride=self.stride)

        # See the autograd section for explanation of what happens here.
        input_im2col = torch.nn.functional.unfold(input.type(torch.float), self.kernel_size, self.dilation, self.padding, self.stride).type(input.type())
        input_transpose = input_im2col.transpose(1, 2)
        input_reshape = input_transpose.reshape(-1, input_transpose.size()[-1])

        weight = self.weight.view(self.weight.size()[0], -1)
        # See the autograd section for explanation of what happens here.
        if self.mode == "fxpfxp":
            mm_out = TLUTLinearFXPFXPFunction.apply(input_reshape, weight, None, 
                self.hwcfg["temporal"], self.widthi, self.widthw, self.hwcfg["widtht"], self.degree, self.delta, self.cycle_act, -self.cycle_act,
                self.hwcfg["rounding"], self.hwcfg["quantilei"], self.hwcfg["quantilew"])
        elif self.mode == "fxpfp":
            mm_out = TLUTLinearFXPFPFunction.apply(input_reshape, weight, None, 
                self.hwcfg["temporal"], self.width, self.hwcfg["widtht"], self.degree, self.delta, self.cycle_act, -self.cycle_act,
                self.hwcfg["rounding"], self.hwcfg["quantilei"], self.hwcfg["quantilew"])
        elif self.mode == "fpfp":
            mm_out = TLUTLinearFPFPFunction.apply(input_reshape, weight, None, 
                self.hwcfg["temporal"], self.width, self.hwcfg["widtht"], self.degree, self.delta, self.cycle_act, -self.cycle_act)
        mm_out_reshape = mm_out.reshape(input.size()[0], -1, mm_out.size()[-1])
        mm_out_transpose = mm_out_reshape.transpose(1, 2)
        output = torch.nn.functional.fold(mm_out_transpose.type(torch.float), output_size, (1, 1)).type(input.type())

        if self.bias is None:
            return output
        else:
            return output + self.bias.view([1, self.bias.size()[0], 1, 1])


class FSUConv2dStoNe(torch.nn.Conv2d):
    """
    This module is the conv2d layer with unary input and unary output, and its API is similar to the Conv2d class, except:
    1) weight_ext: external binary weight
    2) bias_ext: external binary bias
    3) mode: input spike polarity
    4) format: binary weight format
    5) width: binary weight width
    6) scale: accumulation scale
    7) depth: accumulator depth

    The allowed coding for input is rate coding for high accuracy.
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
            "mode" : "bipolar",
            "format" : "fxp",
            "widthw" : 8,
            "scale" : None,
            "depth" : 20,
            "leak" : 0.5,
            "widthg" : 0.1
        }):
        super(FSUConv2dStoNe, self).__init__(in_channels, 
                                                out_channels, 
                                                kernel_size, 
                                                stride, 
                                                padding, 
                                                dilation, 
                                                groups, 
                                                bias, 
                                                padding_mode)
        self.hwcfg = {}
        self.hwcfg["mode"] = hwcfg["mode"].lower()
        self.hwcfg["format"] = hwcfg["format"].lower()
        self.hwcfg["widthw"] = hwcfg["widthw"]
        self.hwcfg["scale"] = hwcfg["scale"]
        self.hwcfg["depth"] = hwcfg["depth"]
        self.hwcfg["leak"] = hwcfg["leak"]
        self.hwcfg["widthg"] = hwcfg["widthg"]

        assert self.hwcfg["format"] in ["fxp", "bfloat16", "float16", "float32"], \
            "Error: the hw config 'format' in " + str(self) + " class requires one of ['fxp', 'bfloat16', 'float16', 'float32']."

        self.mode = hwcfg["mode"].lower()
        assert self.mode in ["unipolar", "bipolar"], \
            "Error: the hw config 'mode' in " + str(self) + " class requires one of ['unipolar', 'bipolar']."

        if self.hwcfg["format"] in ["fxp", "float32"]:
            self.format = torch.float32
        elif self.hwcfg["format"] in ["bfloat16"]:
            self.format = torch.bfloat16
        elif self.hwcfg["format"] in ["float16"]:
            self.format = torch.float16

        if self.hwcfg["format"] in ["fxp"]:
            self.quant = Round(intwidth=self.hwcfg["depth"]-self.hwcfg["widthw"], fracwidth=self.hwcfg["widthw"]-1)

        # define the linear weight and bias
        if weight_ext is not None:
            assert (weight_ext.size()[0], weight_ext.size()[1], weight_ext.size()[2], weight_ext.size()[3]) == (out_channels, in_channels, num2tuple(kernel_size)[0], num2tuple(kernel_size)[1]), \
                "Error: the hw config 'out_channels, in_channels, kernel_size' in " + str(self) + " class unmatches the binary weight shape."
            self.weight.data = weight_ext
        
        assert (bias is False) and (bias_ext is None), "Error: bias=True in " + str(self) + " class is not supported."

        # if bias and (bias_ext is not None):
        #     assert bias_ext.size()[0] == out_channels, \
        #         "Error: the hw config 'out_channels' in " + str(self) + " class unmatches the binary bias shape."
        #     self.bias.data = bias_ext

        self.leak_alpha = self.hwcfg["leak"]
        self.vth = self.hwcfg["scale"]
        if self.hwcfg["mode"] == "unipolar":
            self.m = 1.
            self.k = 0.
        else:
            self.m = 2.
            self.k = 1.

        self.padding = padding
        
        # padding value flag for bipolar
        self.timestep_even = True
        # padding for unipolar
        self.padding_0 = torch.nn.ConstantPad2d(self.padding, 0)
        # padding for bipolar
        self.padding_1 = torch.nn.ConstantPad2d(self.padding, 1)

    def forward_bptt(self, input, u_prev):
        if self.mode == "bipolar" and self.timestep_even is False:
            input_padding = self.padding_1(input)
            self.timestep_even = not self.timestep_even
        else:
            input_padding = self.padding_0(input)

        if self.hwcfg["format"] in ["fxp"]:
            ws = torch.nn.functional.conv2d(input_padding.type(torch.float)*self.m-self.k, self.quant(self.weight.type(torch.float)), self.bias, 
                        self.stride, 0, self.dilation, self.groups)
            ws = self.quant(ws)
        else:
            ws = torch.nn.functional.conv2d(input_padding.type(torch.float)*self.m-self.k, self.weight.type(torch.float), self.bias, 
                        self.stride, 0, self.dilation, self.groups)
        us = self.leak_alpha * u_prev + ws.type(self.format)
        out = NCFireStep.apply(us, self.vth, self.hwcfg["widthg"]).type(self.format)
        u = us - self.vth * (out*self.m-self.k)
        return out, us, u

    def forward(self, input, u_prev):
        return self.forward_bptt(input, u_prev)


class FSUConv2dNC(torch.nn.Conv2d):
    def __init__(
        self, 
        in_channels, 
        out_channels,
        kernel_size,
        bias=False, 
        weight_ext=None, 
        bias_ext=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        hwcfg={
            "widthw" : 8,
            "format" : "float32",
            "scale" : 0.8,
            "depth" : 12,
            "leak" : 0.94,
            "widthg" : 1.25,
            "time_step":10
        }):
        super(FSUConv2dNC, self).__init__(in_channels, out_channels,kernel_size,bias)
        self.hwcfg = {}
        self.hwcfg["widthw"] = hwcfg["widthw"]
        self.hwcfg["format"] = hwcfg["format"]
        self.hwcfg["scale"] = hwcfg["scale"]
        self.hwcfg["depth"] = hwcfg["depth"]
        self.hwcfg["leak"] = hwcfg["leak"]
        self.hwcfg["widthg"] = hwcfg["widthg"]
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        assert self.hwcfg["format"] in ["bfloat16", "float16", "float32", "fxp"], \
            "Error: the hw config 'formatw' in " + str(self) + " class requires one of ['bfloat16', 'float16', 'float32', 'fxp']."
        
        if self.hwcfg["format"] in ["fxp", "float32"]:
            self.format = torch.float32
        elif self.hwcfg["format"] in ["bfloat16"]:
            self.format = torch.bfloat16
        elif self.hwcfg["format"] in ["float16"]:
            self.format = torch.float16

        if self.hwcfg["format"] in ["fxp"]:
            self.quant = Round(intwidth=self.hwcfg["depth"]-self.hwcfg["width"], fracwidth=self.hwcfg["width"]-1)
            
        assert (bias is False) and (bias_ext is None), "Error: bias=True in " + str(self) + " class is not supported."
        
        if weight_ext is not None:
            self.weight.data = weight_ext
        if bias and bias_ext is not None:
            self.bias.data = bias_ext
        
    def forward(self, input, U_prev):
        if self.hwcfg["format"] in ["fxp"]:
            x = torch.nn.functional.conv2d(input.type(torch.float),self.quant(self.weight.type(torch.float)),self.bias,self.stride,
                                          self.padding,self.dilation,self.groups)
            x = self.quant(x)
        else:
            x = torch.nn.functional.conv2d(input.type(torch.float),self.weight.type(torch.float),self.bias,self.stride,
                                          self.padding,self.dilation,self.groups)
        U_s = self.hwcfg["leak"] * U_prev + x.type(self.format)
        output = NCFireStep.apply(U_s, self.hwcfg["scale"], self.hwcfg["widthg"]).type(self.format)
        U = U_s * (1 - output)
        return output, U_s, U

