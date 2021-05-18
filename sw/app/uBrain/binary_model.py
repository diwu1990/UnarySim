import math
import warnings
import numbers
from typing import List, Tuple, Optional, overload, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import dropout


class ScaleReLU(nn.Hardtanh):
    """
    clip the input when it is larger than 1.
    """
    def __init__(self, scale=1., inplace: bool = False):
        super(ScaleReLU, self).__init__(0., scale, inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class ScaleHardsigmoid(nn.Module):
    """
    valid input range is [-1, +1].
    """
    def __init__(self, scale=3):
        super(ScaleHardsigmoid, self).__init__()
        self.scale = scale

    def forward(self, x) -> str:
        return nn.Hardsigmoid()(x * self.scale)


class HardGRUCell(nn.GRUCell):
    """
    This is a GRUCell by replacing sigmoid and tanh with scalehardsigmoid and hardtanh if hard is set to True.
    Batch is always the dim[0].
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, hard: bool = True) -> None:
        super(HardGRUCell, self).__init__(input_size, hidden_size, bias)
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
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        self.check_forward_input(input)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        self.check_forward_hidden(input, hx, '')
        
        gate_i = F.linear(input, self.weight_ih, self.bias_ih)
        gate_h = F.linear(hx, self.weight_hh, self.bias_hh)

        # reset, update, and new gates
        i_r, i_z, i_n = gate_i.chunk(3, 1)
        h_r, h_z, h_n = gate_h.chunk(3, 1)

        resetgate_in = i_r + h_r
        updategate_in = i_z + h_z

        # The resetgate/updategate are scaled additions
        resetgate = self.resetgate_sigmoid(resetgate_in)
        updategate = self.updategate_sigmoid(updategate_in)

        # The newgate is the non-scaled addition of newgate_in inputs (with the product) in unary computing
        newgate_in = i_n + (resetgate * h_n)
        newgate = self.newgate_tanh(newgate_in)

        # this is a MUX function in unary computing
        hy = (1 - updategate) * newgate + updategate * hx

        return hy


class HardGRU(nn.GRU):
    """
    This is a GRU by replacing sigmoid and tanh with hardsigmoid with hardtanh.
    TBD
    """
    def __init__(self, *args, **kwargs):
        super(HardGRU, self).__init__(*args, **kwargs)
        raise NotImplementedError("HardGRU is not implemented yet.")


class Cascade_CNN_RNN_Binary(nn.Module):
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
                    rnn_win_sz=10,
                    rnn_hidden_sz=1024,
                    rnn_hard=True,
                    keep_prob=0.5,
                    num_class=5):
        super(Cascade_CNN_RNN_Binary, self).__init__()
        self.input_sz = input_sz
        self.fc_sz = fc_sz
        self.rnn_win_sz = rnn_win_sz
        self.rnn_hidden_sz = rnn_hidden_sz

        # CNN
        self.conv1          = nn.Conv2d(1        , cnn_chn  , (cnn_kn_sz, cnn_kn_sz), bias=True, padding=cnn_padding)
        self.conv2          = nn.Conv2d(cnn_chn  , cnn_chn*2, (cnn_kn_sz, cnn_kn_sz), bias=True, padding=cnn_padding)
        self.conv3          = nn.Conv2d(cnn_chn*2, cnn_chn*4, (cnn_kn_sz, cnn_kn_sz), bias=True, padding=cnn_padding)
        self.fc4            = nn.Linear((input_sz[0]+3*2*(cnn_padding-1))*(input_sz[1]+3*2*(cnn_padding-1))*cnn_chn*4, fc_sz, bias=True)
        self.fc4_drop       = nn.Dropout(p=1-keep_prob)

        if linear_act.lower() == "scalehardsigmoid":
            self.conv1_act  = ScaleHardsigmoid()
            self.conv2_act  = ScaleHardsigmoid()
            self.conv3_act  = ScaleHardsigmoid()
            self.fc4_act    = ScaleHardsigmoid()
            self.fc7_act    = ScaleHardsigmoid()
        elif linear_act.lower() == "scalerelu":
            self.conv1_act  = ScaleReLU()
            self.conv2_act  = ScaleReLU()
            self.conv3_act  = ScaleReLU()
            self.fc4_act    = ScaleReLU()
            self.fc7_act    = ScaleReLU()
        elif linear_act.lower() == "sigmoid":
            self.conv1_act  = nn.Sigmoid()
            self.conv2_act  = nn.Sigmoid()
            self.conv3_act  = nn.Sigmoid()
            self.fc4_act    = nn.Sigmoid()
            self.fc7_act    = nn.Sigmoid()
        elif linear_act.lower() == "hardtanh":
            self.conv1_act  = nn.Hardtanh()
            self.conv2_act  = nn.Hardtanh()
            self.conv3_act  = nn.Hardtanh()
            self.fc4_act    = nn.Hardtanh()
            self.fc7_act    = nn.Hardtanh()
        elif linear_act.lower() == "tanh":
            self.conv1_act  = nn.Tanh()
            self.conv2_act  = nn.Tanh()
            self.conv3_act  = nn.Tanh()
            self.fc4_act    = nn.Tanh()
            self.fc7_act    = nn.Tanh()
        elif linear_act.lower() == "relu":
            self.conv1_act  = nn.ReLU()
            self.conv2_act  = nn.ReLU()
            self.conv3_act  = nn.ReLU()
            self.fc4_act    = nn.ReLU()
            self.fc7_act    = nn.ReLU()
        elif linear_act.lower() == "relu6":
            self.conv1_act  = nn.ReLU6()
            self.conv2_act  = nn.ReLU6()
            self.conv3_act  = nn.ReLU6()
            self.fc4_act    = nn.ReLU6()
            self.fc7_act    = nn.ReLU6()

        # RNN
        self.grucell6 = HardGRUCell(fc_sz, rnn_hidden_sz, bias=True, hard=rnn_hard)
        # self.gru6 = nn.GRU(fc_sz, rnn_hidden_sz)

        # MLP
        self.fc7 = nn.Linear(rnn_hidden_sz, fc_sz, bias=True)
        self.fc7_drop = nn.Dropout(p=1-keep_prob)
        self.fc8 = nn.Linear(fc_sz, num_class, bias=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, input):
        # CNN
        self.conv1_i        = input.view(-1, 1, self.input_sz[0], self.input_sz[1])
        self.conv1_o        = self.conv1(self.conv1_i)
        self.conv1_act_o    = self.conv1_act(self.conv1_o)
        self.conv2_o        = self.conv2(self.conv1_act_o)
        self.conv2_act_o    = self.conv2_act(self.conv2_o)
        self.conv3_o        = self.conv3(self.conv2_act_o)
        self.conv3_act_o    = self.conv3_act(self.conv3_o)
        self.conv3_act_o    = self.conv3_act_o.view(self.conv3_act_o.shape[0], -1)
        self.fc4_o          = self.fc4(self.conv3_act_o)
        self.fc4_act_o      = self.fc4_act(self.fc4_o)
        self.fc4_act_o      = self.fc4_drop(self.fc4_act_o)
        self.fc4_act_o      = self.fc4_act_o.view(self.rnn_win_sz, -1, self.fc_sz)

        # RNN
        self.rnn_out = []
        hx = torch.zeros(self.fc4_act_o[0].size(0), self.rnn_hidden_sz, dtype=input.dtype, device=input.device)
        for i in range(self.rnn_win_sz):
            hx = self.grucell6(self.fc4_act_o[i], hx)
            self.rnn_out.append(hx)
        # hx = torch.zeros(1, self.fc4_act_o[0].size(0), self.rnn_hidden_sz, dtype=input.dtype, device=input.device)
        # self.rnn_out, _ = self.gru6(self.fc4_act_o, hx)

        # MLP
        self.fc7_i          = self.rnn_out[-1]
        self.fc7_o          = self.fc7(self.fc7_i)
        self.fc7_act_o      = self.fc7_act(self.fc7_o)
        self.fc7_act_o      = self.fc7_drop(self.fc7_act_o)
        self.fc8_o          = self.fc8(self.fc7_act_o)
        output              = self.sm(self.fc8_o)
        return output
