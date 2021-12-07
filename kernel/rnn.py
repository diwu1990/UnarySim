import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from UnarySim.kernel import FSUAdd
from UnarySim.kernel import FSUMul
from UnarySim.kernel import FSULinear
from torch.cuda.amp import autocast
from typing import List, Tuple, Optional, overload, Union
from UnarySim.kernel import HUBHardsigmoid
from UnarySim.kernel import truncated_normal, Round
from UnarySim.stream import BinGen, RNG, BSGen
from UnarySim.metric import ProgError


class FSUMGUCell(torch.nn.Module):
    """
    This is a minimal gated unit with unary computing, corresponding to HardMGUCell with "hard" asserted.
    The scalehardsigmoid is scaled addition (x+1)/2, and hardtanh is direct pass.
    This module follows the uBrain implementation style to maximize hardware reuse.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, 
                    binary_weight_f=None, binary_bias_f=None, binary_weight_n=None, binary_bias_n=None, 
                    hx_buffer=None, 
                    bitwidth=8, mode="bipolar", depth=10, depth_ismul=6) -> None:
        super(FSUMGUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bitwidth = bitwidth

        assert mode=="bipolar", "FSUMGUCell requires 'bipolar' mode."
        assert (binary_weight_f.size()[0], binary_weight_f.size()[1]) == (hidden_size, hidden_size + input_size), "Incorrect weight_f shape."
        assert (binary_weight_n.size()[0], binary_weight_n.size()[1]) == (hidden_size, hidden_size + input_size), "Incorrect weight_n shape."
        if bias is True:
            assert binary_bias_f.size()[0] == hidden_size, "Incorrect bias_f shape."
            assert binary_bias_n.size()[0] == hidden_size, "Incorrect bias_n shape."

        self.fg_ug_tanh = FSULinear(hidden_size + input_size, hidden_size, bias=bias, 
                                            binary_weight=binary_weight_f, binary_bias=binary_bias_f, bitwidth=bitwidth, mode=mode, scaled=False, depth=depth)
        self.ng_ug_tanh = FSULinear(hidden_size + input_size, hidden_size, bias=bias, 
                                            binary_weight=binary_weight_n, binary_bias=binary_bias_n, bitwidth=bitwidth, mode=mode, scaled=False, depth=depth)

        self.fg_sigmoid = FSUAdd(mode=mode, scaled=True, dim=0, depth=depth)
        self.fg_hx_mul = FSUMul(bitwidth=bitwidth, mode=mode, static=True, input_prob_1=hx_buffer)
        self.fg_ng_mul = FSUMul(bitwidth=depth_ismul, mode=mode, static=False)
        self.hy_add = FSUAdd(mode=mode, scaled=False, dim=0, depth=depth, entry=3)

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

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, 
                    binary_weight_f=None, binary_bias_f=None, binary_weight_n=None, binary_bias_n=None, 
                    rng="Sobol", bitwidth=8, mode="bipolar", depth=10, depth_ismul=6) -> None:
        super(HUBMGUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.rng = rng
        self.bitwidth = bitwidth
        self.mode = mode
        self.depth = depth
        self.depth_ismul = depth_ismul

        self.weight_f = binary_weight_f
        self.bias_f = binary_bias_f
        self.weight_n = binary_weight_n
        self.bias_n = binary_bias_n

    @autocast()
    def forward(self, input: Tensor, hx: Tensor) -> Tensor:
        if hx is None:
            hx = torch.zeros(input.size()[0], self.hidden_size, dtype=input.dtype, device=input.device)

        rnncell = FSUMGUCell(self.input_size, self.hidden_size, bias=self.bias, 
                        binary_weight_f=self.weight_f, binary_bias_f=self.bias_f, binary_weight_n=self.weight_n, binary_bias_n=self.bias_n, 
                        hx_buffer=hx, 
                        bitwidth=self.bitwidth, mode=self.mode, depth=self.depth, depth_ismul=self.depth_ismul).to(input.device)
        
        iSource = BinGen(input, bitwidth=self.bitwidth, mode=self.mode)().to(input.device)
        iRNG = RNG(self.bitwidth, 1, self.rng)().to(input.device)
        iBSG = BSGen(iSource, iRNG).to(input.device)
        # iPE = ProgError(input, scale=1, mode=self.mode).to(input.device)

        hSource = BinGen(hx, bitwidth=self.bitwidth, mode=self.mode)().to(input.device)
        hRNG = RNG(self.bitwidth, 1, self.rng)().to(input.device)
        hBSG = BSGen(hSource, hRNG).to(input.device)
        # hPE = ProgError(hx, scale=1, mode=self.mode).to(input.device)

        oPE = ProgError(torch.zeros(input.size()[0], self.hidden_size, dtype=input.dtype, device=input.device), 
                        scale=1, mode=self.mode).to(input.device)

        for c in range(2**self.bitwidth):
            idx = torch.zeros(iSource.size(), dtype=torch.long, device=input.device)
            iBS = iBSG(idx + c)
            # iPE.Monitor(iBS)

            hdx = torch.zeros(hSource.size(), dtype=torch.long, device=input.device)
            hBS = hBSG(hdx + c)
            # hPE.Monitor(hBS)

            oBS = rnncell(iBS, hBS)
            oPE.Monitor(oBS)

        hy = oPE()[0]
        return hy


class HardMGUCell(torch.nn.Module):
    """
    This is a minimal gated unit by replacing sigmoid and tanh with scalehardsigmoid and hardtanh if hard is set to True.
    Batch is always the dim[0].
    Refer to "Simplified Minimal Gated Unit Variations for Recurrent Neural Networks" and "Gate-Variants of Gated Recurrent Unit (GRU) Neural Networks" for more details.
    Hardtanh is to bound data to the legal unary range.
    This module is fully unary computing aware, i.e., all intermediate data are bounded to the legal unary range.
    This module follows the uBrain implementation style to maximize hardware reuse.
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, hard: bool = True) -> None:
        super(HardMGUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.hard = hard
        if hard == True:
            self.fg_sigmoid = HUBHardsigmoid()
            self.ng_tanh = nn.Hardtanh()
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
        self.fg_in = nn.Hardtanh()(F.linear(self.fg_ug_in, self.weight_f, self.bias_f))
        self.fg = self.fg_sigmoid(self.fg_in)
        
        # new gate
        self.fg_hx = self.fg * hx
        self.ng_ug_in = torch.cat((self.fg_hx, input), 1)
        self.ng = self.ng_tanh(F.linear(self.ng_ug_in, self.weight_n, self.bias_n))

        # output
        self.fg_ng = self.fg * self.ng
        self.fg_ng_inv = 0 - self.fg_ng
        hy = nn.Hardtanh()(self.ng + self.fg_ng_inv + self.fg_hx)
        return hy


class HardMGUCellFXP(torch.nn.Module):
    """
    This is a minimal gated unit by replacing sigmoid and tanh with scalehardsigmoid and hardtanh if hard is set to True.
    Batch is always the dim[0].
    Refer to "Simplified Minimal Gated Unit Variations for Recurrent Neural Networks" and "Gate-Variants of Gated Recurrent Unit (GRU) Neural Networks" for more details.
    Hardtanh is to bound data to the legal unary range.
    This module is fully unary computing aware, i.e., all intermediate data are bounded to the legal unary range.
    This module follows the uBrain implementation style to maximize hardware reuse.
    This module applies fixed-point quantization.
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
            self.ng_tanh = nn.Hardtanh()
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
        self.fg_in = nn.Hardtanh()(F.linear(self.trunc(self.fg_ug_in), self.trunc(self.weight_f), self.trunc(self.bias_f)))
        self.fg = self.fg_sigmoid(self.trunc(self.fg_in))
        
        # new gate
        self.fg_hx = self.trunc(self.fg) * self.trunc(hx)
        self.ng_ug_in = torch.cat((self.trunc(self.fg_hx), self.trunc(input)), 1)
        self.ng = self.ng_tanh(self.trunc(F.linear(self.trunc(self.ng_ug_in), self.trunc(self.weight_n), self.trunc(self.bias_n))))

        # output
        self.fg_ng = self.trunc(self.fg) * self.trunc(self.ng)
        self.fg_ng_inv = 0 - self.trunc(self.fg_ng)
        hy = nn.Hardtanh()(self.trunc(self.ng) + self.trunc(self.fg_ng_inv) + self.trunc(self.fg_hx))
        return hy


class HardMGUCellNUA(torch.nn.Module):
    """
    This is a minimal gated unit by replacing sigmoid and tanh with scalehardsigmoid and hardtanh if hard is set to True.
    Batch is always the dim[0].
    Refer to "Simplified Minimal Gated Unit Variations for Recurrent Neural Networks" and "Gate-Variants of Gated Recurrent Unit (GRU) Neural Networks" for more details.
    Hardtanh is to bound data to the legal unary range.
    This module follows the uBrain implementation style to maximize hardware reuse.
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, hard: bool = True) -> None:
        super(HardMGUCellNUA, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.hard = hard
        if hard == True:
            self.fg_sigmoid = HUBHardsigmoid()
            self.ng_tanh = nn.Hardtanh()
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
        self.fg_in = F.linear(self.fg_ug_in, self.weight_f, self.bias_f)
        self.fg = self.fg_sigmoid(self.fg_in)
        
        # new gate
        self.fg_hx = self.fg * hx
        self.ng_ug_in = torch.cat((self.fg_hx, input), 1)
        self.ng_in = F.linear(self.ng_ug_in, self.weight_n, self.bias_n)
        self.ng = self.ng_tanh(self.ng_in)

        # output
        self.fg_ng_inv = (0 - self.fg) * self.ng
        hy = self.ng + self.fg_ng_inv + self.fg_hx

        return hy


class HardMGUCellPT(torch.nn.RNNCellBase):
    """
    This is a minimal gated unit by replacing sigmoid and tanh with scalehardsigmoid and hardtanh if hard is set to True.
    Batch is always the dim[0].
    Refer to "Simplified Minimal Gated Unit Variations for Recurrent Neural Networks" and "Gate-Variants of Gated Recurrent Unit (GRU) Neural Networks" for more details.
    Hardtanh is to bound data to the legal unary range.
    This module is fully unary computing aware, i.e., all intermediate data are bounded to the legal unary range.
    This module follows the PyTorch implementation style (PT).
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, hard: bool = True) -> None:
        super(HardMGUCellPT, self).__init__(input_size, hidden_size, bias, num_chunks=2)
        self.hard = hard
        if hard == True:
            self.forgetgate_sigmoid = HUBHardsigmoid()
            self.newgate_tanh = nn.Hardtanh()
        else:
            self.forgetgate_sigmoid = nn.Sigmoid()
            self.newgate_tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data = truncated_normal(weight, mean=0, std=stdv)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if hx is None:
            hx = torch.zeros(input.size()[0], self.hidden_size, dtype=input.dtype, device=input.device)
        
        self.gate_i = nn.Hardtanh()(F.linear(input, self.weight_ih, self.bias_ih))
        self.gate_h = nn.Hardtanh()(F.linear(hx, self.weight_hh, self.bias_hh))
        self.i_f, self.i_n = self.gate_i.chunk(2, 1)
        self.h_f, self.h_n = self.gate_h.chunk(2, 1)

        self.forgetgate_in = nn.Hardtanh()(self.i_f + self.h_f)
        self.forgetgate = self.forgetgate_sigmoid(self.forgetgate_in)

        # newgate = newgate_tanh(i_n + forgetgate * h_n)
        self.newgate_prod = self.forgetgate * self.h_n
        self.newgate = self.newgate_tanh(self.i_n + self.newgate_prod)

        # hy = (1 - forgetgate) * newgate + forgetgate * hx
        self.forgetgate_inv_prod = (0 - self.forgetgate) * self.newgate
        self.forgetgate_prod = self.forgetgate * hx
        hy = nn.Hardtanh()(self.newgate + self.forgetgate_inv_prod + self.forgetgate_prod)

        return hy


class HardGRUCellNUAPT(torch.nn.RNNCellBase):
    """
    This is a standard GRUCell by replacing sigmoid and tanh with scalehardsigmoid and hardtanh if hard is set to True.
    Batch is always the dim[0].
    This module is not fully unary computing aware (NUA), i.e., not all intermediate data are bounded to the legal unary range.
    This module follows the PyTorch implementation style (PT).
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, hard: bool = True) -> None:
        super(HardGRUCellNUAPT, self).__init__(input_size, hidden_size, bias, num_chunks=3)
        self.hard = hard
        if hard == True:
            self.resetgate_sigmoid = HUBHardsigmoid()
            self.updategate_sigmoid = HUBHardsigmoid()
            self.newgate_tanh = nn.Hardtanh()
        else:
            self.resetgate_sigmoid = nn.Sigmoid()
            self.updategate_sigmoid = nn.Sigmoid()
            self.newgate_tanh = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data = truncated_normal(weight, mean=0, std=stdv)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        if hx is None:
            hx = torch.zeros(input.size()[0], self.hidden_size, dtype=input.dtype, device=input.device)
        
        gate_i = F.linear(input, self.weight_ih, self.bias_ih)
        gate_h = F.linear(hx, self.weight_hh, self.bias_hh)

        i_r, i_z, i_n = gate_i.chunk(3, 1)
        h_r, h_z, h_n = gate_h.chunk(3, 1)

        resetgate_in = i_r + h_r
        updategate_in = i_z + h_z

        resetgate = self.resetgate_sigmoid(resetgate_in)
        updategate = self.updategate_sigmoid(updategate_in)

        newgate_in = i_n + (resetgate * h_n)
        newgate = self.newgate_tanh(newgate_in)

        hy = (1 - updategate) * newgate + updategate * hx

        return hy


