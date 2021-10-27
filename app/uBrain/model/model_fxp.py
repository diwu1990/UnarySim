import math
import warnings
import numbers
from typing import List, Tuple, Optional, overload, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from UnarySim.kernel.sigmoid import ScaleHardsigmoid
from UnarySim.kernel.relu import ScaleReLU
from UnarySim.kernel.utils import truncated_normal
from UnarySim.kernel.rnn import HardMGUCell as HardMGUCell
from UnarySim.kernel.rnn import HardGRUCellNUAPT as HardGRUCell


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
        # self.gate_i = []
        # self.gate_h = []
        # self.forgetgate = []
        # self.newgate_prod = []
        # self.newgate = []
        # self.forgetgate_inv_prod = []
        # self.forgetgate_prod = []
        self.rnn_out = []
        hx = torch.zeros(self.fc3_trans_o[0].size()[0], self.rnn_hidden_sz, dtype=input.dtype, device=input.device)
        for i in range(self.rnn_win_sz):
            hx = self.rnncell4(self.fc3_trans_o[i], hx)
            # self.gate_i.append(self.rnncell4.gate_i)
            # self.gate_h.append(self.rnncell4.gate_h)
            # self.forgetgate.append(self.rnncell4.forgetgate)
            # self.newgate_prod.append(self.rnncell4.newgate_prod)
            # self.newgate.append(self.rnncell4.newgate)
            # self.forgetgate_inv_prod.append(self.rnncell4.forgetgate_inv_prod)
            # self.forgetgate_prod.append(self.rnncell4.forgetgate_prod)
            self.rnn_out.append(hx)

        # MLP
        self.fc5_i          = self.rnn_out[-1]
        self.fc5_o          = self.fc5(self.fc5_i)
        return nn.Hardtanh()(self.fc5_o)

