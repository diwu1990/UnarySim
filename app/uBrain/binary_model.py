import math
import warnings
import numbers
from typing import List, Tuple, Optional, overload, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def truncated_normal(t, mean=0.0, std=0.01):
    torch.nn.init.normal_(t, mean=mean, std=std)
    while True:
        cond = torch.logical_or(t < mean - 2*std, t > mean + 2*std)
        if torch.sum(cond):
            t = torch.where(cond, torch.nn.init.normal_(torch.ones_like(t), mean=mean, std=std), t)
        else:
            break
    return t


class ScaleReLU(torch.nn.Hardtanh):
    """
    clip the input when it is larger than 1.
    """
    def __init__(self, scale=1., inplace: bool = False):
        super(ScaleReLU, self).__init__(0., scale, inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class ScaleHardsigmoid(torch.nn.Module):
    """
    valid input range is [-1, +1].
    """
    def __init__(self, scale=3):
        super(ScaleHardsigmoid, self).__init__()
        self.scale = scale

    def forward(self, x) -> str:
        return nn.Hardsigmoid()(x * self.scale)


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


class HardMGUCell(torch.nn.RNNCellBase):
    """
    This is a minimal gated unit by replacing sigmoid and tanh with scalehardsigmoid and hardtanh if hard is set to True.
    Batch is always the dim[0].
    Refer to "Simplified Minimal Gated Unit Variations for Recurrent Neural Networks" and "Gate-Variants of Gated Recurrent Unit (GRU) Neural Networks" for more details.
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
        
        gate_i = F.linear(input, self.weight_ih, self.bias_ih)
        gate_h = F.linear(hx, self.weight_hh, self.bias_hh)
        i_f, i_n = gate_i.chunk(2, 1)
        h_f, h_n = gate_h.chunk(2, 1)

        forgetgate_in = i_f + h_f
        forgetgate = self.forgetgate_sigmoid(forgetgate_in)

        newgate_prod = forgetgate * h_n
        newgate_in = i_n + newgate_prod
        newgate = self.newgate_tanh(newgate_in)

        hy = (1 - forgetgate) * newgate + forgetgate * hx

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

        self.h_n_hardtanh = nn.Hardtanh()(self.h_n)
        self.newgate_prod = self.forgetgate * self.h_n_hardtanh
        self.i_n_hardtanh = nn.Hardtanh()(self.i_n)
        self.newgate_in = self.i_n_hardtanh + self.newgate_prod
        self.newgate = self.newgate_tanh(self.newgate_in)

        self.forgetgate_inv_prod = (0 - self.forgetgate) * self.newgate
        self.forgetgate_prod = self.forgetgate * hx
        hy = self.newgate + self.forgetgate_inv_prod + self.forgetgate_prod

        return hy


class Cascade_CNN_RNN_Binary(torch.nn.Module):
    """
    This is the binary version of the cascade CNN RNN for BCI
    """
    def __init__(self,
                    input_sz=(10, 11),
                    linear_act="hardtanh",
                    cnn_chn=32,
                    cnn_kn_sz=3,
                    cnn_padding=0,
                    fc_sz=1024,
                    rnn="gru",
                    rnn_win_sz=10,
                    rnn_hidden_sz=1024,
                    rnn_hard=True,
                    bias=True,
                    init_std=None,
                    keep_prob=0.5,
                    num_class=5):
        super(Cascade_CNN_RNN_Binary, self).__init__()
        self.input_sz = input_sz
        self.fc_sz = fc_sz
        self.rnn_win_sz = rnn_win_sz
        self.rnn_hidden_sz = rnn_hidden_sz
        self.bias = bias
        self.init_std = init_std
        # CNN
        self.conv1          = nn.Conv2d(1        , cnn_chn  , (cnn_kn_sz, cnn_kn_sz), bias=bias, padding=cnn_padding)
        self.conv2          = nn.Conv2d(cnn_chn  , cnn_chn*2, (cnn_kn_sz, cnn_kn_sz), bias=bias, padding=cnn_padding)
        self.fc3            = nn.Linear((input_sz[0]+2*2*(cnn_padding-1))*(input_sz[1]+2*2*(cnn_padding-1))*cnn_chn*2, fc_sz, bias=bias)
        self.fc3_drop       = nn.Dropout(p=1-keep_prob)

        # RNN
        if rnn.lower() == "gru":
            self.rnncell4 = HardGRUCell(fc_sz, rnn_hidden_sz, bias=bias, hard=rnn_hard)
        elif rnn.lower() == "mgu":
            self.rnncell4 = HardMGUCell(fc_sz, rnn_hidden_sz, bias=bias, hard=rnn_hard)

        # MLP
        self.fc5 = nn.Linear(rnn_hidden_sz, num_class, bias=bias)

        self.linear_act = linear_act.lower()

        if self.linear_act == "scalehardsigmoid":
            self.conv1_act  = ScaleHardsigmoid()
            self.conv2_act  = ScaleHardsigmoid()
            self.fc3_act    = ScaleHardsigmoid()
        elif self.linear_act == "scalerelu":
            self.conv1_act  = ScaleReLU()
            self.conv2_act  = ScaleReLU()
            self.fc3_act    = ScaleReLU()
        elif self.linear_act == "sigmoid":
            self.conv1_act  = nn.Sigmoid()
            self.conv2_act  = nn.Sigmoid()
            self.fc3_act    = nn.Sigmoid()
        elif self.linear_act == "hardtanh":
            self.conv1_act  = nn.Hardtanh()
            self.conv2_act  = nn.Hardtanh()
            self.fc3_act    = nn.Hardtanh()
        elif self.linear_act == "tanh":
            self.conv1_act  = nn.Tanh()
            self.conv2_act  = nn.Tanh()
            self.fc3_act    = nn.Tanh()
        elif self.linear_act == "relu":
            self.conv1_act  = nn.ReLU()
            self.conv2_act  = nn.ReLU()
            self.fc3_act    = nn.ReLU()
        elif self.linear_act == "relu6":
            self.conv1_act  = nn.ReLU6()
            self.conv2_act  = nn.ReLU6()
            self.fc3_act    = nn.ReLU6()
        elif self.linear_act == "elu":
            self.conv1_act  = nn.ELU()
            self.conv2_act  = nn.ELU()
            self.fc3_act    = nn.ELU()

        self.init_weight()

    def init_weight(self):
        if self.init_std is None:
            if self.linear_act == "hardtanh":
                std = 0.035
            elif self.linear_act == "scalerelu":
                std = 0.05
            else:
                std = 0.1
        else:
            std = self.init_std
        self.conv1.weight.data  = truncated_normal(self.conv1.weight, mean=0, std=std)
        self.conv2.weight.data  = truncated_normal(self.conv2.weight, mean=0, std=std)
        self.fc3.weight.data    = truncated_normal(self.fc3.weight,   mean=0, std=std)
        self.fc5.weight.data    = truncated_normal(self.fc5.weight,   mean=0, std=std)

        if self.bias == True:
            self.conv1.bias.data.fill_(0.1)
            self.conv2.bias.data.fill_(0.1)
            self.fc3.bias.data.fill_(0.1)
            self.fc5.bias.data.fill_(0.1)

    def forward(self, input):
        # CNN
        self.conv1_i        = input.view(-1, 1, self.input_sz[0], self.input_sz[1])
        self.conv1_o        = self.conv1(self.conv1_i)
        self.conv1_act_o    = self.conv1_act(self.conv1_o)
        self.conv2_o        = self.conv2(self.conv1_act_o)
        self.conv2_act_o    = self.conv2_act(self.conv2_o)
        self.fc3_i          = self.conv2_act_o.view(self.conv2_act_o.shape[0], -1)
        self.fc3_o          = self.fc3(self.fc3_i)
        self.fc3_act_o      = self.fc3_act(self.fc3_o)
        self.fc3_drop_o     = self.fc3_drop(self.fc3_act_o)
        self.fc3_view_o     = self.fc3_drop_o.view(-1, self.rnn_win_sz, self.fc_sz)
        self.fc3_trans_o    = self.fc3_view_o.transpose(0, 1)

        # RNN
        self.rnn_out = []
        hx = torch.zeros(self.fc3_trans_o[0].size(0), self.rnn_hidden_sz, dtype=input.dtype, device=input.device)
        for i in range(self.rnn_win_sz):
            hx = self.rnncell4(self.fc3_trans_o[i], hx)
            self.rnn_out.append(hx)

        # MLP
        self.fc5_i          = self.rnn_out[-1]
        self.fc5_o          = self.fc5(self.fc5_i)
        return nn.Hardtanh()(self.fc5_o)


def print_tensor_unary_outlier(tensor, name):
    min = tensor.min().item()
    max = tensor.max().item()
    outlier = torch.sum(torch.gt(tensor, 1)) + torch.sum(torch.lt(tensor, -1))
    outlier_ratio = outlier / torch.prod(torch.tensor(tensor.size()))
    print("{:20s}".format(name) + \
            ": min:" + "{:10f}".format(min) + \
            "; max:" + "{:10f}".format(max) + \
            "; outlier:" + "{:10f} %".format(outlier_ratio * 100))