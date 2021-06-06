import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from UnarySim.kernel.add import FSUAdd
from UnarySim.kernel.mul import FSUMul
from UnarySim.kernel.linear import FSULinearPC
from torch.cuda.amp import autocast
from typing import List, Tuple, Optional, overload, Union
from UnarySim.kernel.sigmoid import ScaleHardsigmoid
from UnarySim.kernel.utils import truncated_normal


class FSUMGUCell(torch.nn.Module):
    """
    This is a minimal gated unit with unary computing.
    The scalehardsigmoid is scaled addition (x+1)/2, and hardtanh is direct pass.
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, 
                    binary_weight_ih=None, binary_bias_ih=None, binary_weight_hh=None, binary_bias_hh=None, 
                    bitwidth=8, mode="bipolar", depth=10) -> None:
        super(FSUMGUCell, self).__init__()
        self.bitwidth = bitwidth
        assert mode=="bipolar", "Unsupported mode in FSUMGUCell."

        self.forgetgate_in_add = FSUAdd(mode=mode, scaled=False, dim=0, depth=depth, entry=input_size+hidden_size+bias+bias)
        self.forgetgate_sigmoid = FSUAdd(mode=mode, scaled=True, dim=0, depth=depth)
        self.h_n_acc = FSUAdd(mode=mode, scaled=False, dim=0, depth=depth, entry=hidden_size+bias)
        self.newgate_in_mul = FSUMul(bitwidth=4, mode=mode, static=False)
        self.i_n_acc = FSUAdd(mode=mode, scaled=False, dim=0, depth=depth, entry=input_size+bias)
        self.newgate_in_add = FSUAdd(mode=mode, scaled=False, dim=0, depth=depth, entry=2)
        self.newgate_tanh = nn.Identity()
        self.hy_inv_mul = FSUMul(bitwidth=4, mode=mode, static=False)
        self.hy_mul = FSUMul(bitwidth=4, mode=mode, static=False)
        self.hy_add = FSUAdd(mode=mode, scaled=False, dim=0, depth=depth, entry=3)

        self.gate_i_linearpc = FSULinearPC(2*hidden_size, input_size,  bias=bias, 
                                            binary_weight=binary_weight_ih, binary_bias=binary_bias_ih, bitwidth=bitwidth, mode=mode)
        self.gate_h_linearpc = FSULinearPC(2*hidden_size, hidden_size, bias=bias, 
                                            binary_weight=binary_weight_hh, binary_bias=binary_bias_hh, bitwidth=bitwidth, mode=mode)

    @autocast()
    def forward(self, input: Tensor, hx: Tensor) -> Tensor:
        self.gate_i = self.gate_i_linearpc(input)
        self.gate_h = self.gate_h_linearpc(hx)
        self.i_f, self.i_n = self.gate_i.chunk(2, 1)
        self.h_f, self.h_n = self.gate_h.chunk(2, 1)

        # The forgetgate is a scaled addition
        self.forgetgate_in = self.forgetgate_in_add(torch.stack([self.i_f, self.h_f], dim=0))
        self.forgetgate = self.forgetgate_sigmoid(torch.stack([self.forgetgate_in, torch.ones_like(self.forgetgate_in)], dim=0))

        # The newgate is the non-scaled addition of newgate_in inputs (with the product) in unary computing
        self.h_n_hardtanh = self.h_n_acc(self.h_n.unsqueeze(0))
        self.newgate_prod = self.newgate_in_mul(self.forgetgate, self.h_n_hardtanh)
        self.i_n_hardtanh = self.i_n_acc(self.i_n.unsqueeze(0))
        self.newgate_in = self.newgate_in_add(torch.stack([self.i_n_hardtanh, self.newgate_prod], dim=0))
        self.newgate = self.newgate_tanh(self.newgate_in)

        self.forgetgate_inv_prod = self.hy_inv_mul(1 - self.forgetgate, self.newgate)
        self.forgetgate_prod = self.hy_mul(self.forgetgate, hx)
        hy = self.hy_add(torch.stack([self.newgate, self.forgetgate_inv_prod, self.forgetgate_prod], dim=0))

        return hy


class HardMGUCell(torch.nn.RNNCellBase):
    """
    This is a minimal gated unit by replacing sigmoid and tanh with scalehardsigmoid and hardtanh if hard is set to True.
    Batch is always the dim[0].
    Refer to "Simplified Minimal Gated Unit Variations for Recurrent Neural Networks" and "Gate-Variants of Gated Recurrent Unit (GRU) Neural Networks" for more details.
    Hardtanh is to bound data to the legal unary range.
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, hard: bool = True) -> None:
        super(HardMGUCell, self).__init__(input_size, hidden_size, bias, num_chunks=2)
        self.hard = hard
        if hard == True:
            self.forgetgate_sigmoid = ScaleHardsigmoid()
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
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        self.check_forward_hidden(input, hx, '')
        
        self.gate_i = F.linear(input, self.weight_ih, self.bias_ih)
        self.gate_h = F.linear(hx, self.weight_hh, self.bias_hh)
        self.i_f, self.i_n = self.gate_i.chunk(2, 1)
        self.h_f, self.h_n = self.gate_h.chunk(2, 1)

        self.forgetgate_in = self.i_f + self.h_f
        self.forgetgate = self.forgetgate_sigmoid(self.forgetgate_in)

        # newgate = newgate_tanh(forgetgate * h_n + newgate_prod)
        self.h_n_hardtanh = nn.Hardtanh()(self.h_n)
        self.newgate_prod = self.forgetgate * self.h_n_hardtanh
        self.i_n_hardtanh = nn.Hardtanh()(self.i_n)
        self.newgate_in = self.i_n_hardtanh + self.newgate_prod
        self.newgate = self.newgate_tanh(self.newgate_in)

        # hy = (1 - forgetgate) * newgate + forgetgate * hx
        self.forgetgate_inv_prod = (0 - self.forgetgate) * self.newgate
        self.forgetgate_prod = self.forgetgate * hx
        hy = self.newgate + self.forgetgate_inv_prod + self.forgetgate_prod

        return hy


class HardGRUCell(torch.nn.RNNCellBase):
    """
    This is a standard GRUCell by replacing sigmoid and tanh with scalehardsigmoid and hardtanh if hard is set to True.
    Batch is always the dim[0].
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, hard: bool = True) -> None:
        super(HardGRUCell, self).__init__(input_size, hidden_size, bias, num_chunks=3)
        self.hard = hard
        if hard == True:
            self.resetgate_sigmoid = ScaleHardsigmoid()
            self.updategate_sigmoid = ScaleHardsigmoid()
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
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        self.check_forward_hidden(input, hx, '')
        
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