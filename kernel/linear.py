import torch
import math
import copy
from UnarySim.stream import RNG, BinGen, BSGen
from UnarySim.kernel import FSUAdd, rshift_offset
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

    The allowed coding for input, weight and bias with guaranteed accuracy can have the following three options.s
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
    The allowed coding for input, weight and bias with guaranteed accuracy can have the following three options.s
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
        
        self.width = hwcfg["width"]
        self.mode = hwcfg["mode"].lower()
        assert self.mode in ["unipolar", "bipolar"], \
            "Error: the hw config 'mode' in " + str(self) + " class requires one of ['unipolar', 'bipolar']."

        self.btype = swcfg["btype"]
        self.rtype = swcfg["rtype"]
        self.stype = swcfg["stype"]

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
            # RNG for bias, should always apply rate coding
            hwcfg_brng = {
                "width" : hwcfg["width"],
                "rng" : "sobol",
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
        obit_i1 = torch.empty(0, device=input.device)
        torch.matmul(ibit_i1, wbit_i1.transpose(1, 2), out=obit_i1)
        obit_i1.squeeze_(1)
        
        if self.has_bias is True:
            bbit = self.bbsg(self.brdx).type(torch.float)
            self.brdx.add_(1)
            obit_i1 += bbit.unsqueeze(0).expand_as(obit_i1)

        if self.mode == "unipolar":
            return obit_i1
        
        if self.mode == "bipolar":
            # generate weight and bias bits for current cycle
            wbit_i0 = 1 - self.wbsg_i0(self.wrdx_i0).type(torch.float)
            if wbit_i0.size()[0] != batch:
                wbit_i0 = torch.cat(batch*[wbit_i0], 0)
                self.wrdx_i0 = torch.cat(batch*[self.wrdx_i0], 0)
            torch.add(self.wrdx_i0, 1 - input.unsqueeze(1).type(torch.long), out=self.wrdx_i0)
            
            ibit_i0 = 1 - ibit_i1
            obit_i0 = torch.empty(0, device=input.device)
            torch.matmul(ibit_i0, wbit_i0.transpose(1, 2), out=obit_i0)
            obit_i0.squeeze_(1)

            return obit_i1 + obit_i0
    
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
        obit_i1 = torch.empty(0, device=input.device)
        torch.matmul(ibit_i1, wbit_i1.transpose(1, 2), out=obit_i1)
        obit_i1.squeeze_(1)
        
        if self.has_bias is True:
            bbit = self.bbsg(self.brdx).type(torch.float)
            self.brdx.add_(1)
            obit_i1 += bbit.unsqueeze(0).expand_as(obit_i1)

        if self.mode == "unipolar":
            return obit_i1
        
        if self.mode == "bipolar":
            # generate weight and bias bits for current cycle
            wbit_i0 = 1 - wbit_i1
            ibit_i0 = 1 - ibit_i1
            obit_i0 = torch.empty(0, device=input.device)
            torch.matmul(ibit_i0, wbit_i0.transpose(1, 2), out=obit_i0)
            obit_i0.squeeze_(1)

            return obit_i1 + obit_i0

    @autocast()
    def forward(self, input):
        assert len(input.size()) == 2, \
            "Error: the input of the " + str(self) + " class needs 2 dimensions."
        if self.wtc:
            return self.FSULinear_PC_wtc(input).type(self.stype)
        else:
            return self.FSULinear_PC_wrc(input).type(self.stype)
        

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

        assert not (self.hwcfg["rngi"] in ["race", "tc", "race10", "tc10"] and self.hwcfg["rngw"] in ["race", "tc", "race10", "tc10"]), \
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
        
        print(self.hwcfg)

        # weight and bias
        if weight_ext is not None:
            self.weight.data = weight_ext
        
        if bias and (bias_ext is not None):
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
        
        # generate the value map for mul using current rng
        # dim 0 is input index
        # the tensor input value is the actual value produced by the irng
        self.imap = torch.nn.Parameter(torch.empty(self.cycle_max), requires_grad=False)
        cycle_ival = torch.empty(0)
        torch.cat(self.cycle_max*[torch.arange(self.cycle_max, dtype=torch.float).unsqueeze(1)], 1, out=cycle_ival)
        cycle_ibit = torch.empty(0)
        torch.gt(cycle_ival, self.irng.unsqueeze(0), out=cycle_ibit)
        self.imap.data = torch.sum(cycle_ibit, 1).squeeze_().type(torch.long)

        # dim 0 is input index, dim 1 is weight index
        # the tensor value is the actual weight value produced by the wrng, under a specific input and weight
        self.wmap = torch.nn.Parameter(torch.empty(self.cycle_max, self.cycle_max), requires_grad=False)
        cycle_wbit = torch.empty(0)
        torch.gt(cycle_ival, self.wrng.unsqueeze(0), out=cycle_wbit)
        for c in range(self.cycle_max):
            self.wmap.data[c] = torch.sum(cycle_wbit[:, 0:self.imap.data[c]], 1).squeeze_()
        
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
                                        self.cycle_act, self.wmap)

    
# Inherit from Function
class HUBLinearFunction(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, 
                rshift_i=3, rshift_w=3, rshift_o=3, 
                cycle=128, wmap=None):
        ctx.save_for_backward(input, weight, bias)

        assert len(input.size()) == 2, \
            "Error: the input of HUBLinearFunction class needs 2 dimensions."

        # first dim should always be batch
        batch = input.size()[0]
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # input preparation
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # scale input to range 0~2^bitwidth-1
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
        # scale weight with batch to range 0~2^bitwidth-1
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
        torch.mul(wmap[buf_i, buf_w], act_wght, out=act_wght)
        
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
            self.weight.data = weight_ext
        
        if bias and (bias_ext is not None):
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