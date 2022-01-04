import torch
import math
import copy
from UnarySim.stream import RNG, BinGen, BSGen
from UnarySim.kernel import FSUAdd, rshift_offset, Round, NCFireStep
from torch.cuda.amp import autocast

class FSULinear(torch.nn.Module):
    """
    This module is the fully connected layer with unary input and unary output, and its API is similar to the Linear class (input/output feature count, bias flag), except:
    1) weight_ext: external binary weight
    2) bias_ext: external binary bias
    3) width: binary data width
    4) mode: unary data mode
    5) scale: accumulation scale
    6) depth: accumulator depth
    7) rng: weight rng type
    8) dimr: weight rng dimension

    The allowed coding for input, weight and bias with guaranteed accuracy can have the following three options.
    (input, weight, bias):
    1) rate, rate, rate
    2) rate, temporal, rate
    3) temporal, rate, rate
    However, this module itself does not force the input coding. Thus, above coding constraints should be done by users.
    """
    def __init__(
        self, 
        in_features, 
        out_features, 
        bias=True, 
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
        super(FSULinear, self).__init__()
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

        if (hwcfg["mode"].lower() == "bipolar") and (hwcfg["scale"] is not None) and (hwcfg["scale"] != (in_features + bias)):
            assert self.hwcfg["rng"].lower() not in ["race", "tc", "race10", "tc10"], \
                "Error: the hw config 'rng' in " + str(self) + " class should avoid ['race', 'tc', 'race10', 'tc10'] for bipolar data with non-scaled addition."

        assert self.swcfg["btype"] == torch.float, \
            "Error: the sw config 'btype' in " + str(self) + " class requires 'torch.float'."
        assert self.swcfg["rtype"] == torch.float, \
            "Error: the sw config 'rtype' in " + str(self) + " class requires 'torch.float'."
        assert self.swcfg["stype"] == torch.float, \
            "Error: the sw config 'stype' in " + str(self) + " class requires 'torch.float'."

        self.PC = FSULinearPC(
            in_features, 
            out_features, 
            bias=bias, 
            weight_ext=weight_ext, 
            bias_ext=bias_ext, 
            hwcfg=self.hwcfg,
            swcfg=self.swcfg)

        self.scale = hwcfg["scale"]
        if self.scale is None:
            scale_add = in_features + bias
        else:
            scale_add = self.scale
        hwcfg_acc = copy.deepcopy(self.hwcfg)
        hwcfg_acc["scale"] = scale_add
        hwcfg_acc["entry"] = in_features + bias
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


class FSULinearPC(torch.nn.Linear):
    """
    This module is the parallel counter result of FSULinear before generating the bitstreams.
    The allowed coding for input, weight and bias with guaranteed accuracy can have the following three options.
    (input, weight, bias):
    1) rate, rate, rate
    2) rate, temporal, rate
    3) temporal, rate, rate
    However, this module itself does not force the input coding. Thus, above coding constraints should be done by users.
    """
    def __init__(
        self, 
        in_features, 
        out_features, 
        bias=True, 
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
        super(FSULinearPC, self).__init__(in_features, out_features, bias=bias)
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
            assert (weight_ext.size()[0], weight_ext.size()[1]) == (out_features, in_features), \
                "Error: the hw config 'out_features, in_features' in " + str(self) + " class unmatches the binary weight shape."
            self.weight.data = BinGen(weight_ext, self.hwcfg, self.swcfg)()
        
        if bias and (bias_ext is not None):
            assert bias_ext.size()[0] == out_features, \
                "Error: the hw config 'out_features' in " + str(self) + " class unmatches the binary bias shape."
            self.bias.data = BinGen(bias_ext, self.hwcfg, self.swcfg)()
            # RNG for bias, same as RNG for weight
            hwcfg_brng = {
                "width" : hwcfg["width"],
                "rng" : hwcfg["rng"],
                "dimr" : hwcfg["dimr"]
            }
            self.brng = RNG(hwcfg_brng, swcfg)()

        # define the kernel linear for input bit 1
        self.wbsg_i1 = BSGen(self.weight, self.wrng, swcfg)
        self.wrdx_i1 = torch.nn.Parameter(torch.zeros_like(self.weight, dtype=torch.long), requires_grad=False).unsqueeze(0)
        if self.has_bias is True:
            self.bbsg = BSGen(self.bias, self.brng, swcfg)
            self.brdx = torch.nn.Parameter(torch.zeros_like(self.bias, dtype=torch.long), requires_grad=False)
        
        # if bipolar, define a kernel for input bit 0, note that there is no bias required for this kernel
        if (self.mode == "bipolar") and (self.wtc is False):
            self.wbsg_i0 = BSGen(self.weight, self.wrng, swcfg)
            self.wrdx_i0 = torch.nn.Parameter(torch.zeros_like(self.weight, dtype=torch.long), requires_grad=False).unsqueeze(0)

    def FSULinear_PC_wrc(self, input):
        # this function is for weight with rate coding
        # first dim should always be batch
        batch = input.size()[0]

        # generate weight and bias bits for current cycle
        wbit_i1 = self.wbsg_i1(self.wrdx_i1).type(torch.float)
        if wbit_i1.size()[0] != batch:
            wbit_i1 = torch.cat(batch*[wbit_i1], 0)
            self.wrdx_i1 = torch.cat(batch*[self.wrdx_i1], 0)
        torch.add(self.wrdx_i1, input.unsqueeze(1).type(torch.long), out=self.wrdx_i1)
        
        ibit_i1 = input.unsqueeze(1).type(torch.float)
        obin_i1 = torch.empty(0, device=input.device)
        torch.matmul(ibit_i1, wbit_i1.transpose(1, 2), out=obin_i1)
        obin_i1.squeeze_(1)
        
        if self.has_bias is True:
            bbit = self.bbsg(self.brdx).type(torch.float)
            self.brdx.add_(1)
            obin_i1 += bbit.unsqueeze(0).expand_as(obin_i1)

        if self.mode == "unipolar":
            return obin_i1
        
        if self.mode == "bipolar":
            # generate weight and bias bits for current cycle
            wbit_i0 = 1 - self.wbsg_i0(self.wrdx_i0).type(torch.float)
            if wbit_i0.size()[0] != batch:
                wbit_i0 = torch.cat(batch*[wbit_i0], 0)
                self.wrdx_i0 = torch.cat(batch*[self.wrdx_i0], 0)
            torch.add(self.wrdx_i0, 1 - input.unsqueeze(1).type(torch.long), out=self.wrdx_i0)
            
            ibit_i0 = 1 - ibit_i1
            obin_i0 = torch.empty(0, device=input.device)
            torch.matmul(ibit_i0, wbit_i0.transpose(1, 2), out=obin_i0)
            obin_i0.squeeze_(1)

            return obin_i1 + obin_i0
    
    def FSULinear_PC_wtc(self, input):
        # this function is for weight with temporal coding
        # first dim should always be batch
        batch = input.size()[0]

        # generate weight and bias bits for current cycle
        wbit_i1 = self.wbsg_i1(self.wrdx_i1).type(torch.float)
        if wbit_i1.size()[0] != batch:
            wbit_i1 = torch.cat(batch*[wbit_i1], 0)
            self.wrdx_i1 = torch.cat(batch*[self.wrdx_i1], 0)
        torch.add(self.wrdx_i1, torch.ones_like(input).unsqueeze(1).type(torch.long), out=self.wrdx_i1)
        
        ibit_i1 = input.unsqueeze(1).type(torch.float)
        obin_i1 = torch.empty(0, device=input.device)
        torch.matmul(ibit_i1, wbit_i1.transpose(1, 2), out=obin_i1)
        obin_i1.squeeze_(1)
        
        if self.has_bias is True:
            bbit = self.bbsg(self.brdx).type(torch.float)
            self.brdx.add_(1)
            obin_i1 += bbit.unsqueeze(0).expand_as(obin_i1)

        if self.mode == "unipolar":
            return obin_i1
        
        if self.mode == "bipolar":
            # generate weight and bias bits for current cycle
            wbit_i0 = 1 - wbit_i1
            ibit_i0 = 1 - ibit_i1
            obin_i0 = torch.empty(0, device=input.device)
            torch.matmul(ibit_i0, wbit_i0.transpose(1, 2), out=obin_i0)
            obin_i0.squeeze_(1)

            return obin_i1 + obin_i0

    @autocast()
    def forward(self, input):
        assert len(input.size()) == 2, \
            "Error: the input of the " + str(self) + " class needs 2 dimensions."
        if self.wtc:
            return self.FSULinear_PC_wtc(input).type(self.swcfg["stype"])
        else:
            return self.FSULinear_PC_wrc(input).type(self.swcfg["stype"])
        

# the HUBLinear and HUBLinearFunction are parallel implementations
class HUBLinear(torch.nn.Linear):
    """
    This module is the fully connected layer for binary signed data in fxp format using unary computing.
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
        in_features, 
        out_features, 
        bias=True, 
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
            "rounding" : "round",
            "signmag" : True
        }):
        super(HUBLinear, self).__init__(in_features, out_features, bias)
        self.hwcfg = {}
        self.hwcfg["widthi"] = hwcfg["widthi"]
        self.hwcfg["rngi"] = hwcfg["rngi"].lower()
        self.hwcfg["quantilei"] = hwcfg["quantilei"]
        self.hwcfg["widthw"] = hwcfg["widthw"]
        self.hwcfg["rngw"] = hwcfg["rngw"].lower()
        self.hwcfg["quantilew"] = hwcfg["quantilew"]
        self.hwcfg["rounding"] = hwcfg["rounding"].lower()
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
        if self.hwcfg["cycle"] is None:
            self.cycle_act = self.cycle_max
        else:
            self.cycle_act = min(self.hwcfg["cycle"], self.cycle_max)
        self.hwcfg["cycle"] = self.cycle_act
        
        # weight and bias
        if weight_ext is not None:
            assert (weight_ext.size()[0], weight_ext.size()[1]) == (out_features, in_features), \
                "Error: the hw config 'out_features, in_features' in " + str(self) + " class unmatches the binary weight shape."
            self.weight.data = weight_ext
        
        if bias and (bias_ext is not None):
            assert bias_ext.size()[0] == out_features, \
                "Error: the hw config 'out_features' in " + str(self) + " class unmatches the binary bias shape."
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
        
        return HUBLinearFunction.apply(input, self.weight, self.bias, 
                                        self.rshift_i, self.rshift_w, self.rshift_o, 
                                        self.cycle_act, self.mapctlee)


# Inherit from Function
class HUBLinearFunction(torch.autograd.Function):
    """
    This code is for rate coding for both input and weight.
    """
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, 
                rshift_i=3, rshift_w=3, rshift_o=3, 
                cycle=128, mapcbsg=None):
        ctx.save_for_backward(input, weight, bias)

        assert len(input.size()) == 2, \
            "Error: the input of HUBLinearFunction class needs 2 dimensions."

        # first dim should always be batch
        batch = input.size()[0]
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # input preparation
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # scale input to range 0~2^widthi-1
        buf_i = torch.empty(0, dtype=torch.long, device=input.device)
        torch.abs((input >> rshift_i).unsqueeze_(1).round().type(torch.long), out=buf_i)
        torch.clamp(buf_i, 0, cycle-1, out=buf_i)
        
        # actual input: its sign
        act_input = torch.empty(0, device=input.device)
        torch.sign(input, out=act_input)
        act_input.unsqueeze_(1)
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # weight preparation
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # scale weight with batch to range 0~2^widthw-1
        buf_w_no_batch = torch.empty(0, dtype=torch.long, device=weight.device)
        torch.abs((weight >> rshift_w).unsqueeze_(0).round().type(torch.long), out=buf_w_no_batch)
        torch.clamp(buf_w_no_batch, 0, cycle-1, out=buf_w_no_batch)
        buf_w = torch.empty(0, dtype=torch.long, device=weight.device)
        torch.cat(batch*[buf_w_no_batch], 0, out=buf_w)

        # get actual weight for calculation
        sign_wght_no_batch = torch.empty(0, device=weight.device)
        torch.sign(weight, out=sign_wght_no_batch)
        sign_wght_no_batch.unsqueeze_(0)
        act_wght = torch.empty(0, device=weight.device)
        torch.cat(batch*[sign_wght_no_batch], 0, out=act_wght)
        torch.mul(mapcbsg[buf_i, buf_w], act_wght, out=act_wght)
        
        output = torch.empty(0, device=weight.device)
        torch.matmul(act_input, act_wght.transpose(1, 2), out=output)
        
        output = (output >> rshift_o).squeeze_(1)
        
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None

    
class FXPLinear(torch.nn.Linear):
    """
    This module is the fully connected layer for binary signed data in fxp format using binary computing.
    The hardware configuration specifies 
    1) the data with in bit for input and weight/bias
    2) the quantile to quantize input and weight/bias
    3) the rounding mode for both input and weight/bias
    """
    def __init__(
        self, 
        in_features, 
        out_features, 
        bias=True, 
        weight_ext=None, 
        bias_ext=None, 
        hwcfg={
            "widthi" : 8,
            "quantilei" : 1,
            "widthw" : 8,
            "quantilew" : 1,
            "rounding" : "round"
        }):
        super(FXPLinear, self).__init__(in_features, out_features, bias)
        self.hwcfg = {}
        self.hwcfg["widthi"] = hwcfg["widthi"]
        self.hwcfg["quantilei"] = hwcfg["quantilei"]
        self.hwcfg["widthw"] = hwcfg["widthw"]
        self.hwcfg["quantilew"] = hwcfg["quantilew"]
        self.hwcfg["rounding"] = hwcfg["rounding"].lower()
        
        # weight and bias
        if weight_ext is not None:
            assert (weight_ext.size()[0], weight_ext.size()[1]) == (out_features, in_features), \
                "Error: the hw config 'out_features, in_features' in " + str(self) + " class unmatches the binary weight shape."
            self.weight.data = weight_ext
        
        if bias and (bias_ext is not None):
            assert bias_ext.size()[0] == out_features, \
                "Error: the hw config 'out_features' in " + str(self) + " class unmatches the binary bias shape."
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
        
        return FXPLinearFunction.apply(input, self.weight, self.bias, self.rshift_i, self.rshift_w, self.rshift_o, self.max_abs_i, self.max_abs_w)

    
# Inherit from Function
class FXPLinearFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, 
                rshift_i=3, 
                rshift_w=3, 
                rshift_o=3, 
                max_abs_i=128, 
                max_abs_w=128):
        ctx.save_for_backward(input, weight, bias)
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # input preparation
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # round input to (bot, top)
        bot_i = 1 - max_abs_i
        top_i = max_abs_i - 1
        i_round = torch.empty(0, device=input.device)
        torch.round(input >> rshift_i, out=i_round)
        torch.clamp(i_round.unsqueeze_(1), bot_i, top_i, out=i_round)
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # weight preparation
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # round input to (bot, top)
        bot_w = 1 - max_abs_w
        top_w = max_abs_w - 1
        w_round = torch.empty(0, device=input.device)
        torch.round(weight >> rshift_w, out=w_round)
        torch.clamp(w_round.unsqueeze_(0), bot_w, top_w, out=w_round)
        
        output = torch.empty(0, device=weight.device)
        torch.matmul(i_round, w_round.transpose(1, 2), out=output)
        output = (output >> rshift_o).squeeze_(1)
        
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class TLUTLinear(torch.nn.Linear):
    """
    This module is the fully connected layer using temporal look-up table (T-LUT). 
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
        in_features, 
        out_features, 
        bias=True, 
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
        super(TLUTLinear, self).__init__(in_features, out_features, bias)
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
            assert (weight_ext.size()[0], weight_ext.size()[1]) == (out_features, in_features), \
                "Error: the hw config 'out_features, in_features' in " + str(self) + " class unmatches the binary weight shape."
            self.weight.data = weight_ext
        
        if bias and (bias_ext is not None):
            assert bias_ext.size()[0] == out_features, \
                "Error: the hw config 'out_features' in " + str(self) + " class unmatches the binary bias shape."
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
        # See the autograd section for explanation of what happens here.
        if self.mode == "fxpfxp":
            return TLUTLinearFXPFXPFunction.apply(input, self.weight, self.bias, 
                self.hwcfg["temporal"], self.widthi, self.widthw, self.hwcfg["widtht"], self.degree, self.delta, self.cycle_act, -self.cycle_act,
                self.hwcfg["rounding"], self.hwcfg["quantilei"], self.hwcfg["quantilew"])
        elif self.mode == "fxpfp":
            return TLUTLinearFXPFPFunction.apply(input, self.weight, self.bias, 
                self.hwcfg["temporal"], self.width, self.hwcfg["widtht"], self.degree, self.delta, self.cycle_act, -self.cycle_act,
                self.hwcfg["rounding"], self.hwcfg["quantilei"], self.hwcfg["quantilew"])
        elif self.mode == "fpfp":
            return TLUTLinearFPFPFunction.apply(input, self.weight, self.bias, 
                self.hwcfg["temporal"], self.width, self.hwcfg["widtht"], self.degree, self.delta, self.cycle_act, -self.cycle_act)
    
# Inherit from Function
class TLUTLinearFXPFXPFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, 
                temporal="i", widthi=8, widthw=8, widtht=4, degree=2, delta=0, cycle_pos=16, cycle_neg=-16, rounding="round", quantilei=1, quantilew=1):
        ctx.save_for_backward(input, weight, bias)

        input_fp32 = input.detach().clone().to(torch.float)
        weight_fp32 = weight.detach().clone().to(torch.float)
        
        input_new = torch.zeros_like(input_fp32)
        weight_new = torch.zeros_like(weight_fp32)

        rshift_i, rshift_w, _ = rshift_offset(input_fp32, weight_fp32, widthi, widthw, rounding, quantilei, quantilew)
        
        torch.trunc((input_fp32 >> rshift_i).clamp(-2**widthi+1, 2**widthi-1), out=input_fp32)
        torch.trunc((weight_fp32 >> rshift_w).clamp(-2**widthw+1, 2**widthw-1), out=weight_fp32)

        if temporal in ["i", "input"]:
            frac = torch.zeros_like(input_fp32)
            for i in range(degree):
                input_fp32 = input_fp32 >> widtht
                torch.frac(input_fp32, out=frac)
                torch.trunc(input_fp32, out=input_fp32)
                torch.clamp(frac << widtht, cycle_neg+1, cycle_pos-1, out=frac)
                torch.add(frac >> widtht, input_new >> widtht, out=input_new)
            input_new = (input_new << (delta + widthi + rshift_i)).type(weight.type())
            weight_new = (weight_fp32 << rshift_w).type(weight.type())
        elif temporal in ["w", "weight"]:
            frac = torch.zeros_like(weight_fp32)
            for i in range(degree):
                weight_fp32 = weight_fp32 >> widtht
                torch.frac(weight_fp32, out=frac)
                torch.trunc(weight_fp32, out=weight_fp32)
                torch.clamp(frac << widtht, cycle_neg+1, cycle_pos-1, out=frac)
                torch.add(frac >> widtht, weight_new >> widtht, out=weight_new)
            input_new = (input_fp32 << rshift_i).type(input.type())
            weight_new = (weight_new << (delta + widthw + rshift_w)).type(input.type())
        
        output = torch.matmul(input_new, weight_new.t())
        
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None


# Inherit from Function
class TLUTLinearFXPFPFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, 
                temporal="i", width=8, widtht=4, degree=2, delta=0, cycle_pos=16, cycle_neg=-16, rounding="round", quantilei=1, quantilew=1):
        ctx.save_for_backward(input, weight, bias)
        input_fp32 = input.detach().clone().to(torch.float)
        weight_fp32 = weight.detach().clone().to(torch.float)

        rshift_i, rshift_w, _ = rshift_offset(input_fp32, weight_fp32, width, width, rounding, quantilei, quantilew)

        if temporal in ["i", "input"]:
            input_new = torch.zeros_like(input_fp32)
            frac = torch.zeros_like(input_fp32)
            torch.trunc((input_fp32 >> rshift_i).clamp(-2**width+1, 2**width-1), out=input_fp32)
            for i in range(degree):
                input_fp32 = input_fp32 >> widtht
                torch.frac(input_fp32, out=frac)
                torch.trunc(input_fp32, out=input_fp32)
                torch.clamp(frac << widtht, cycle_neg+1, cycle_pos-1, out=frac)
                torch.add(frac >> widtht, input_new >> widtht, out=input_new)
            input_new = (input_new << (delta + width + rshift_i)).type(weight.type())
            weight_new = weight
        elif temporal in ["w", "weight"]:
            weight_new = torch.zeros_like(weight_fp32)
            frac = torch.zeros_like(weight_fp32)
            torch.trunc((weight_fp32 >> rshift_w).clamp(-2**width+1, 2**width-1), out=weight_fp32)
            for i in range(degree):
                weight_fp32 = weight_fp32 >> widtht
                torch.frac(weight_fp32, out=frac)
                torch.trunc(weight_fp32, out=weight_fp32)
                torch.clamp(frac << widtht, cycle_neg+1, cycle_pos-1, out=frac)
                torch.add(frac >> widtht, weight_new >> widtht, out=weight_new)
            input_new = input
            weight_new = (weight_new << (delta + width + rshift_w)).type(input.type())
        
        output = torch.matmul(input_new, weight_new.t())
        
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None


# Inherit from Function
class TLUTLinearFPFPFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, 
                temporal="i", width=8, widtht=4, degree=2, delta=0, cycle_pos=16, cycle_neg=-16):
        ctx.save_for_backward(input, weight, bias)

        dtype = input.type()

        if temporal in ["i", "input"]:
            input_fp32 = input.detach().clone().type(torch.float)
            mantissa, exponent = torch.frexp(input_fp32)
            frac = torch.zeros_like(input_fp32)
            mantissa_new = torch.zeros_like(input_fp32)
        elif temporal in ["w", "weight"]:
            weight_fp32 = weight.detach().clone().type(torch.float)
            mantissa, exponent = torch.frexp(weight_fp32)
            frac = torch.zeros_like(weight_fp32)
            mantissa_new = torch.zeros_like(weight_fp32)

        mantissa = mantissa << width
        for i in range(degree):
            mantissa = mantissa >> widtht
            torch.frac(mantissa, out=frac)
            torch.trunc(mantissa, out=mantissa)
            torch.clamp(frac << widtht, cycle_neg+1, cycle_pos-1, out=frac)
            torch.add(frac >> widtht, mantissa_new >> widtht, out=mantissa_new)

        mantissa_new = mantissa_new << delta

        if temporal in ["i", "input"]:
            input_new = torch.ldexp(mantissa_new, exponent).type(dtype)
            weight_new = weight
        elif temporal in ["w", "weight"]:
            input_new = input
            weight_new = torch.ldexp(mantissa_new, exponent).type(dtype)
        
        output = torch.matmul(input_new, weight_new.t())

        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None


class FSULinearStoNe(torch.nn.Linear):
    """
    This module is the fully connected layer with unary input and unary output, and its API is similar to the Linear class (input/output feature count, bias flag), except:
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
        in_features, 
        out_features, 
        bias=False, 
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
        super(FSULinearStoNe, self).__init__(in_features, out_features, bias)
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
            assert (weight_ext.size()[0], weight_ext.size()[1]) == (out_features, in_features), \
                "Error: the hw config 'out_features, in_features' in " + str(self) + " class unmatches the binary weight shape."
            self.weight.data = weight_ext
        
        assert (bias is False) and (bias_ext is None), "Error: bias=True in " + str(self) + " class is not supported."

        # if bias and (bias_ext is not None):
        #     assert bias_ext.size()[0] == out_features, \
        #         "Error: the hw config 'out_features' in " + str(self) + " class unmatches the binary bias shape."
        #     self.bias.data = bias_ext
        
        self.leak_alpha = self.hwcfg["leak"]
        self.vth = self.hwcfg["scale"]
        if self.hwcfg["mode"] == "unipolar":
            self.m = 1.
            self.k = 0.
        else:
            self.m = 2.
            self.k = 1.

    def forward_bptt(self, input, u_prev):
        if self.hwcfg["format"] in ["fxp"]:
            ws = torch.matmul(self.quant(input).type(self.format)*self.m-self.k, self.quant(self.weight).t().type(self.format))
            ws = self.quant(ws)
        else:
            ws = torch.matmul(input.type(self.format)*self.m-self.k, self.weight.t().type(self.format))
        us = self.leak_alpha * u_prev + ws
        out = NCFireStep.apply(us, self.vth, self.hwcfg["widthg"]).type(self.format)
        u = us - self.vth * (out*self.m-self.k)
        return out, us, u

    def forward(self, input, u_prev):
        return self.forward_bptt(input, u_prev)

