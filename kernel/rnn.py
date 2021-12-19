import math
import torch
import copy
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from UnarySim.kernel import FSUAdd
from UnarySim.kernel import FSUMul
from UnarySim.kernel import FSULinear
from torch.cuda.amp import autocast
from typing import List, Tuple, Optional, overload, Union
from UnarySim.kernel import HUBHardsigmoid, HUBHardtanh
from UnarySim.kernel import truncated_normal, Round
from UnarySim.stream import BinGen, RNG, BSGen
from UnarySim.metric import ProgError


class FSUMGUCell(torch.nn.Module):
    """
    This is a minimal gated unit with unary computing, corresponding to HardMGUCell with "hard" asserted.
    The scalehardsigmoid is scaled addition (x+1)/2, and hardtanh is direct pass.
    This module follows the uBrain implementation style to maximize hardware reuse.
    """

    def __init__(
        self, 
        input_size: int, hidden_size: int, bias: bool = True, 
        weight_ext_f=None, bias_ext_f=None, weight_ext_n=None, bias_ext_n=None, 
        hx_buffer=None, 
        hwcfg={
            "width" : 8,
            "mode" : "bipolar",
            "depth" : 10,
            "depth_ismul" : 6,
            "rng" : "Sobol",
            "dimr" : 1
        },
        swcfg={
            "btype" : torch.float,
            "rtype" : torch.float,
            "stype" : torch.float
        }) -> None:
        super(FSUMGUCell, self).__init__()
        self.hwcfg = {}
        self.hwcfg["width"] = hwcfg["width"]
        self.hwcfg["mode"] = hwcfg["mode"].lower()
        self.hwcfg["depth"] = hwcfg["depth"]
        self.hwcfg["depth_ismul"] = hwcfg["depth_ismul"]
        self.hwcfg["rng"] = hwcfg["rng"].lower()
        self.hwcfg["dimr"] = hwcfg["dimr"]

        self.swcfg = {}
        self.swcfg["btype"] = swcfg["btype"]
        self.swcfg["rtype"] = swcfg["rtype"]
        self.swcfg["stype"] = swcfg["stype"]

        self.input_size = input_size
        self.hidden_size = hidden_size

        assert self.hwcfg["mode"] in ["bipolar"], \
            "Error: the hw config 'mode' in " + str(self) + " class requires 'bipolar'."

        assert (weight_ext_f.size()[0], weight_ext_f.size()[1]) == (hidden_size, hidden_size + input_size), "Incorrect weight_f shape."
        assert (weight_ext_n.size()[0], weight_ext_n.size()[1]) == (hidden_size, hidden_size + input_size), "Incorrect weight_n shape."
        if bias is True:
            assert bias_ext_f.size()[0] == hidden_size, "Incorrect bias_f shape."
            assert bias_ext_n.size()[0] == hidden_size, "Incorrect bias_n shape."

        hwcfg_linear={
            "width" : self.hwcfg["width"],
            "mode" : self.hwcfg["mode"],
            "scale" : 1,
            "depth" : self.hwcfg["depth"],
            "rng" : self.hwcfg["rng"],
            "dimr" : self.hwcfg["dimr"]
        }
        self.fg_ug_tanh = FSULinear(hidden_size + input_size, hidden_size, bias=bias, 
                                            weight_ext=weight_ext_f, bias_ext=bias_ext_f, 
                                            hwcfg=hwcfg_linear, swcfg=swcfg)
        self.ng_ug_tanh = FSULinear(hidden_size + input_size, hidden_size, bias=bias, 
                                            weight_ext=weight_ext_n, bias_ext=bias_ext_n, 
                                            hwcfg=hwcfg_linear, swcfg=swcfg)
        hwcfg_sigm={
            "mode" : self.hwcfg["mode"],
            "scale" : None,
            "dima" : 0,
            "depth" : self.hwcfg["depth"],
            "entry" : None
        }
        self.fg_sigmoid = FSUAdd(hwcfg_sigm, swcfg)
        hwcfg_hx_mul={
            "width" : self.hwcfg["width"],
            "mode" : self.hwcfg["mode"],
            "static" : True,
            "rng" : self.hwcfg["rng"],
            "dimr" : self.hwcfg["dimr"]
        }
        self.fg_hx_mul = FSUMul(in_1_prob=hx_buffer, hwcfg=hwcfg_hx_mul, swcfg=swcfg)
        hwcfg_ng_mul={
            "width" : self.hwcfg["depth_ismul"],
            "mode" : self.hwcfg["mode"],
            "static" : False,
            "rng" : self.hwcfg["rng"],
            "dimr" : self.hwcfg["dimr"]
        }
        self.fg_ng_mul = FSUMul(in_1_prob=None, hwcfg=hwcfg_ng_mul, swcfg=swcfg)
        hwcfg_hy={
            "mode" : self.hwcfg["mode"],
            "scale" : 1,
            "dima" : 0,
            "depth" : self.hwcfg["depth"],
            "entry" : 3
        }
        self.hy_add = FSUAdd(hwcfg_hy, swcfg)

    def check_forward_input(self, input: Tensor) -> None:
        if input.size(1) != self.input_size:
            raise RuntimeError("input has inconsistent input_size: got {}, expected {}".format(input.size(1), self.input_size))

    def check_forward_hidden(self, input: Tensor, hx: Tensor, hidden_label: str = '') -> None:
        if input.size(0) != hx.size(0):
            raise RuntimeError("Input batch size {} doesn't match hidden{} batch size {}".format(input.size(0), hidden_label, hx.size(0)))
        if hx.size(1) != self.hidden_size:
            raise RuntimeError("hidden{} has inconsistent hidden_size: got {}, expected {}".format(hidden_label, hx.size(1), self.hidden_size))

    @autocast()
    def forward(self, input: Tensor, hx: Tensor) -> Tensor:
        self.check_forward_input(input)
        self.check_forward_hidden(input, hx, '')

        # forget gate
        self.fg_ug_in = torch.cat((hx, input), 1)
        self.fg_in = self.fg_ug_tanh(self.fg_ug_in)
        self.fg = self.fg_sigmoid(torch.stack([self.fg_in, torch.ones_like(self.fg_in)], dim=0))
        
        # new gate
        self.fg_hx = self.fg_hx_mul(self.fg)
        self.ng_ug_in = torch.cat((self.fg_hx, input), 1)
        self.ng = self.ng_ug_tanh(self.ng_ug_in)

        # output
        self.fg_ng = self.fg_ng_mul(self.fg, self.ng)
        self.fg_ng_inv = 1 - self.fg_ng
        hy = self.hy_add(torch.stack([self.ng, self.fg_ng_inv, self.fg_hx], dim=0))
        return hy


class HUBMGUCell(torch.nn.Module):
    """
    This is a minimal gated unit with hybrid unary binary computing, corresponding to HardMGUCell with "hard" asserted.
    The scalehardsigmoid is scaled addition (x+1)/2, and hardtanh is direct pass.
    This module follows the uBrain implementation style to maximize hardware reuse.
    """

    def __init__(
        self, input_size: int, hidden_size: int, bias: bool = True, 
        weight_ext_f=None, bias_ext_f=None, weight_ext_n=None, bias_ext_n=None, 
        hwcfg={
            "width" : 8,
            "mode" : "bipolar",
            "depth" : 10,
            "depth_ismul" : 6,
            "rng" : "Sobol",
            "dimr" : 1
        }) -> None:
        super(HUBMGUCell, self).__init__()
        self.hwcfg = {}
        self.hwcfg["width"] = hwcfg["width"]
        self.hwcfg["mode"] = hwcfg["mode"].lower()
        self.hwcfg["depth"] = hwcfg["depth"]
        self.hwcfg["depth_ismul"] = hwcfg["depth_ismul"]
        self.hwcfg["rng"] = hwcfg["rng"].lower()
        self.hwcfg["dimr"] = hwcfg["dimr"]

        self.swcfg = {}
        self.swcfg["btype"] = torch.float
        self.swcfg["rtype"] = torch.float
        self.swcfg["stype"] = torch.float

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.weight_f = weight_ext_f
        self.bias_f = bias_ext_f
        self.weight_n = weight_ext_n
        self.bias_n = bias_ext_n

        self.hwcfg_ope = copy.deepcopy(self.hwcfg)
        self.hwcfg_ope["scale"] = 1

    @autocast()
    def forward(self, input: Tensor, hx: Tensor) -> Tensor:
        if hx is None:
            hx = torch.zeros(input.size()[0], self.hidden_size, dtype=input.dtype, device=input.device)

        rnncell = FSUMGUCell(self.input_size, self.hidden_size, bias=self.bias, 
                        weight_ext_f=self.weight_f, bias_ext_f=self.bias_f, weight_ext_n=self.weight_n, bias_ext_n=self.bias_n, 
                        hx_buffer=hx, 
                        hwcfg=self.hwcfg, swcfg=self.swcfg).to(input.device)
        
        iSource = BinGen(input, self.hwcfg, self.swcfg)().to(input.device)
        iRNG = RNG(self.hwcfg, self.swcfg)().to(input.device)
        iBSG = BSGen(iSource, iRNG, self.swcfg).to(input.device)

        hSource = BinGen(hx, self.hwcfg, self.swcfg)().to(input.device)
        hRNG = RNG(self.hwcfg, self.swcfg)().to(input.device)
        hBSG = BSGen(hSource, hRNG, self.swcfg).to(input.device)

        oPE = ProgError(torch.zeros(input.size()[0], self.hidden_size, dtype=input.dtype, device=input.device), 
                        self.hwcfg_ope).to(input.device)

        for c in range(2**self.hwcfg["width"]):
            idx = torch.zeros(iSource.size(), dtype=torch.long, device=input.device)
            iBS = iBSG(idx + c)

            hdx = torch.zeros(hSource.size(), dtype=torch.long, device=input.device)
            hBS = hBSG(hdx + c)

            oBS = rnncell(iBS, hBS)
            oPE.Monitor(oBS)

        hy = oPE()[0]
        return hy


class HardMGUCell(torch.nn.Module):
    """
    This is a minimal gated unit by replacing sigmoid and tanh with hubhardsigmoid and hubhardtanh if hard is set to True.
    Refer to "Simplified Minimal Gated Unit Variations for Recurrent Neural Networks" and "Gate-Variants of Gated Recurrent Unit (GRU) Neural Networks" for more details.
    This module is fully unary computing aware, i.e., all intermediate data are bounded to the legal unary range.
    This module follows the uBrain implementation style to maximize hardware reuse.
    This modeule assigns batch to dim[0].
    This module applies floating-point data.
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, hard: bool = True) -> None:
        super(HardMGUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.hard = hard
        if hard == True:
            self.fg_sigmoid = HUBHardsigmoid()
            self.ng_tanh = HUBHardtanh()
        else:
            self.fg_sigmoid = nn.Sigmoid()
            self.ng_tanh = nn.Tanh()

        self.weight_f = nn.Parameter(torch.empty((hidden_size, hidden_size + input_size)))
        self.weight_n = nn.Parameter(torch.empty((hidden_size, hidden_size + input_size)))
        if bias:
            self.bias_f = nn.Parameter(torch.empty(hidden_size))
            self.bias_n = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter('bias_f', None)
            self.register_parameter('bias_n', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data = truncated_normal(weight, mean=0, std=stdv)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if hx is None:
            hx = torch.zeros(input.size()[0], self.hidden_size, dtype=input.dtype, device=input.device)
        
        # forget gate
        self.fg_ug_in = torch.cat((hx, input), 1)
        self.fg_in = HUBHardtanh()(F.linear(self.fg_ug_in, self.weight_f, self.bias_f))
        self.fg = self.fg_sigmoid(self.fg_in)
        
        # new gate
        self.fg_hx = self.fg * hx
        self.ng_ug_in = torch.cat((self.fg_hx, input), 1)
        self.ng = self.ng_tanh(F.linear(self.ng_ug_in, self.weight_n, self.bias_n))

        # output
        self.fg_ng = self.fg * self.ng
        self.fg_ng_inv = 0 - self.fg_ng
        hy = HUBHardtanh()(self.ng + self.fg_ng_inv + self.fg_hx)
        return hy


class HardMGUCellFXP(torch.nn.Module):
    """
    This is a minimal gated unit by replacing sigmoid and tanh with hubhardsigmoid and hubhardtanh if hard is set to True.
    Refer to "Simplified Minimal Gated Unit Variations for Recurrent Neural Networks" and "Gate-Variants of Gated Recurrent Unit (GRU) Neural Networks" for more details.
    This module is fully unary computing aware, i.e., all intermediate data are bounded to the legal unary range.
    This module follows the uBrain implementation style to maximize hardware reuse.
    This modeule assigns batch to dim[0].
    This module applies fixed-point quantization using 'intwidth' and 'fracwidth'.
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, hard: bool = True,
                intwidth=3, fracwidth=4) -> None:
        super(HardMGUCellFXP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.hard = hard
        self.trunc = Round(intwidth=intwidth, fracwidth=fracwidth)

        if hard == True:
            self.fg_sigmoid = HUBHardsigmoid()
            self.ng_tanh = HUBHardtanh()
        else:
            self.fg_sigmoid = nn.Sigmoid()
            self.ng_tanh = nn.Tanh()

        self.weight_f = nn.Parameter(torch.empty((hidden_size, hidden_size + input_size)))
        self.weight_n = nn.Parameter(torch.empty((hidden_size, hidden_size + input_size)))
        if bias:
            self.bias_f = nn.Parameter(torch.empty(hidden_size))
            self.bias_n = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter('bias_f', None)
            self.register_parameter('bias_n', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data = truncated_normal(weight, mean=0, std=stdv)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if hx is None:
            hx = torch.zeros(input.size()[0], self.hidden_size, dtype=input.dtype, device=input.device)
        
        # forget gate
        self.fg_ug_in = torch.cat((self.trunc(hx), self.trunc(input)), 1)
        self.fg_in = HUBHardtanh()(F.linear(self.trunc(self.fg_ug_in), self.trunc(self.weight_f), self.trunc(self.bias_f)))
        self.fg = self.fg_sigmoid(self.trunc(self.fg_in))
        
        # new gate
        self.fg_hx = self.trunc(self.fg) * self.trunc(hx)
        self.ng_ug_in = torch.cat((self.trunc(self.fg_hx), self.trunc(input)), 1)
        self.ng = self.ng_tanh(self.trunc(F.linear(self.trunc(self.ng_ug_in), self.trunc(self.weight_n), self.trunc(self.bias_n))))

        # output
        self.fg_ng = self.trunc(self.fg) * self.trunc(self.ng)
        self.fg_ng_inv = 0 - self.trunc(self.fg_ng)
        hy = HUBHardtanh()(self.trunc(self.ng) + self.trunc(self.fg_ng_inv) + self.trunc(self.fg_hx))
        return hy

