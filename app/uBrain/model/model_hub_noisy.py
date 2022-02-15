import math
import warnings
import numbers
from typing import List, Tuple, Optional, overload, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from UnarySim.kernel.conv import HUBConv2d
from UnarySim.kernel.linear import HUBLinear
from UnarySim.kernel.sigmoid import ScaleHardsigmoid
from UnarySim.kernel.relu import ScaleReLU
from UnarySim.kernel.rnn import HUBMGUCell, HardMGUCell
from UnarySim.metric.metric import SourceGen, RNG, BSGen, ProgError
from UnarySim.kernel.utils import progerror_report


class Cascade_CNN_RNN(torch.nn.Module):
    """
    This is the hybrid unary binary version of the cascade CNN RNN for BCI, i.e., uBrain
    """
    def __init__(self, 
                    input_sz=[10, 11],
                    linear_act="scalerelu",
                    cnn_chn=16,
                    cnn_kn_sz=3,
                    cnn_padding=1, # default perform same conv
                    fc_sz=256,
                    rnn="mgu",
                    rnn_win_sz=10,
                    rnn_hidden_sz=64,
                    rnn_hard=True,
                    bias=False,
                    init_std=None,
                    keep_prob=0.5,
                    num_class=[5, 2],

                    bitwidth_tc=8, 
                    bitwidth_rc=8, 
                    rng="Sobol", 
                    conv1_weight=None, 
                    conv1_bias=None, 
                    conv2_weight=None, 
                    conv2_bias=None, 
                    fc3_weight=None, 
                    fc3_bias=None, 
                    rnn4_weight_f=None, 
                    rnn4_bias_f=None, 
                    rnn4_weight_n=None, 
                    rnn4_bias_n=None, 
                    fc5_weight=None, 
                    fc5_bias=None, 
                    depth=10,
                    depth_ismul=5):
        super(Cascade_CNN_RNN, self).__init__()
        self.input_sz = input_sz
        self.cnn_chn = cnn_chn
        self.cnn_kn_sz = cnn_kn_sz
        self.cnn_padding = cnn_padding
        self.fc_sz = fc_sz
        self.rnn_win_sz = rnn_win_sz
        self.rnn_hidden_sz = rnn_hidden_sz
        self.bias = bias
        self.num_class = num_class
        self.bitwidth_tc = bitwidth_tc
        self.bitwidth_rc = bitwidth_rc
        self.rng = rng
        self.conv1_weight = conv1_weight
        self.conv1_bias = conv1_bias
        self.conv2_weight = conv2_weight
        self.conv2_bias = conv2_bias
        self.fc3_weight = fc3_weight
        self.fc3_bias = fc3_bias
        self.rnn4_weight_f = rnn4_weight_f
        self.rnn4_bias_f = rnn4_bias_f
        self.rnn4_weight_n = rnn4_weight_n
        self.rnn4_bias_n = rnn4_bias_n
        self.fc5_weight = fc5_weight
        self.fc5_bias = fc5_bias
        self.cycle_tc = 2**(bitwidth_tc-1)
        self.mode = "bipolar"

        # CNN
        self.conv1          = HUBConv2d(1        , cnn_chn  , (cnn_kn_sz, cnn_kn_sz), bias=bias, padding=cnn_padding, 
                                        binary_weight=self.conv1_weight, binary_bias=self.conv1_bias, rng=self.rng, cycle=self.cycle_tc)
        self.conv2          = HUBConv2d(cnn_chn  , cnn_chn*2, (cnn_kn_sz, cnn_kn_sz), bias=bias, padding=cnn_padding, 
                                        binary_weight=self.conv2_weight, binary_bias=self.conv2_bias, rng=self.rng, cycle=self.cycle_tc)
        self.fc3            = HUBLinear((input_sz[0]+2*2*(cnn_padding-1))*(input_sz[1]+2*2*(cnn_padding-1))*cnn_chn*2, fc_sz, bias=bias, 
                                        binary_weight=self.fc3_weight, binary_bias=self.fc3_bias, rng=self.rng, cycle=self.cycle_tc)
        self.fc3_drop       = nn.Dropout(p=1-keep_prob)

        # RNN
        if rnn.lower() == "mgu":
            self.rnncell4 = HUBMGUCell(fc_sz, rnn_hidden_sz, bias=bias, 
                                binary_weight_f=self.rnn4_weight_f, binary_bias_f=self.rnn4_bias_f, binary_weight_n=self.rnn4_weight_n, binary_bias_n=self.rnn4_bias_n, 
                                rng=rng, bitwidth=bitwidth_rc, mode=self.mode, depth=depth, depth_ismul=depth_ismul)
        else:
            print("rnn type needs to be 'mgu'.")

        # MLP
        self.fc5            = HUBLinear(rnn_hidden_sz, sum(num_class), bias=bias, 
                                        binary_weight=self.fc5_weight, binary_bias=self.fc5_bias, rng=self.rng, cycle=self.cycle_tc)

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

    def forward(self, input, binary_fm_dict=None):
        # input is (batch, win, h, w)

        # CNN
        # added an error term to input in order to mimic the ATC inaccuracy
        self.conv1_i_        = input.view(-1, 1, self.input_sz[0], self.input_sz[1])
        self.conv1_i        = (self.conv1_i_ * (1 + torch.rand(self.conv1_i_.size(), device=self.conv1_i_.device) / 20))
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
        hx = torch.zeros(self.fc3_trans_o[0].size()[0], self.rnn_hidden_sz, dtype=input.dtype, device=input.device)
        for i in range(self.rnn_win_sz):
            hx = self.rnncell4(self.fc3_trans_o[i], hx)
            self.rnn_out.append(hx)

        # MLP
        self.fc5_i          = self.rnn_out[-1]
        self.fc5_o          = self.fc5(self.fc5_i)
        return nn.Hardtanh()(self.fc5_o)



class Cascade_CNN_RNN_fp_rnn(torch.nn.Module):
    """
    This is the hybrid unary binary version of the cascade CNN RNN for BCI, i.e., uBrain
    But the rnn is in fp format, so that entire model is trainable.
    """
    def __init__(self, 
                    input_sz=[10, 11],
                    linear_act="scalerelu",
                    cnn_chn=16,
                    cnn_kn_sz=3,
                    cnn_padding=1, # default perform same conv
                    fc_sz=256,
                    rnn="mgu",
                    rnn_win_sz=10,
                    rnn_hidden_sz=64,
                    rnn_hard=True,
                    bias=False,
                    init_std=None,
                    keep_prob=0.5,
                    num_class=[5, 2],

                    bitwidth_tc=8, 
                    bitwidth_rc=8, 
                    rng="Sobol", 
                    conv1_weight=None, 
                    conv1_bias=None, 
                    conv2_weight=None, 
                    conv2_bias=None, 
                    fc3_weight=None, 
                    fc3_bias=None, 
                    rnn4_weight_f=None, 
                    rnn4_bias_f=None, 
                    rnn4_weight_n=None, 
                    rnn4_bias_n=None, 
                    fc5_weight=None, 
                    fc5_bias=None, 
                    depth=10,
                    depth_ismul=5):
        super(Cascade_CNN_RNN, self).__init__()
        self.input_sz = input_sz
        self.cnn_chn = cnn_chn
        self.cnn_kn_sz = cnn_kn_sz
        self.cnn_padding = cnn_padding
        self.fc_sz = fc_sz
        self.rnn_win_sz = rnn_win_sz
        self.rnn_hidden_sz = rnn_hidden_sz
        self.bias = bias
        self.num_class = num_class
        self.bitwidth_tc = bitwidth_tc
        self.bitwidth_rc = bitwidth_rc
        self.rng = rng
        self.conv1_weight = conv1_weight
        self.conv1_bias = conv1_bias
        self.conv2_weight = conv2_weight
        self.conv2_bias = conv2_bias
        self.fc3_weight = fc3_weight
        self.fc3_bias = fc3_bias
        self.rnn4_weight_f = rnn4_weight_f
        self.rnn4_bias_f = rnn4_bias_f
        self.rnn4_weight_n = rnn4_weight_n
        self.rnn4_bias_n = rnn4_bias_n
        self.fc5_weight = fc5_weight
        self.fc5_bias = fc5_bias
        self.cycle_tc = 2**(bitwidth_tc-1)
        self.mode = "bipolar"

        # CNN
        self.conv1          = HUBConv2d(1        , cnn_chn  , (cnn_kn_sz, cnn_kn_sz), bias=bias, padding=cnn_padding, 
                                        binary_weight=self.conv1_weight, binary_bias=self.conv1_bias, rng=self.rng, cycle=self.cycle_tc)
        self.conv2          = HUBConv2d(cnn_chn  , cnn_chn*2, (cnn_kn_sz, cnn_kn_sz), bias=bias, padding=cnn_padding, 
                                        binary_weight=self.conv2_weight, binary_bias=self.conv2_bias, rng=self.rng, cycle=self.cycle_tc)
        self.fc3            = HUBLinear((input_sz[0]+2*2*(cnn_padding-1))*(input_sz[1]+2*2*(cnn_padding-1))*cnn_chn*2, fc_sz, bias=bias, 
                                        binary_weight=self.fc3_weight, binary_bias=self.fc3_bias, rng=self.rng, cycle=self.cycle_tc)
        self.fc3_drop       = nn.Dropout(p=1-keep_prob)

        # RNN
        if rnn.lower() == "mgu":
            self.rnncell4 = HardMGUCell(fc_sz, rnn_hidden_sz, bias=bias, hard=rnn_hard)
        else:
            print("rnn type needs to be 'mgu'.")

        # MLP
        self.fc5            = HUBLinear(rnn_hidden_sz, sum(num_class), bias=bias, 
                                        binary_weight=self.fc5_weight, binary_bias=self.fc5_bias, rng=self.rng, cycle=self.cycle_tc)

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

    def forward(self, input, binary_fm_dict=None):
        # input is (batch, win, h, w)

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
        hx = torch.zeros(self.fc3_trans_o[0].size()[0], self.rnn_hidden_sz, dtype=input.dtype, device=input.device)
        for i in range(self.rnn_win_sz):
            hx = self.rnncell4(self.fc3_trans_o[i], hx)
            self.rnn_out.append(hx)

        # MLP
        self.fc5_i          = self.rnn_out[-1]
        self.fc5_o          = self.fc5(self.fc5_i)
        return nn.Hardtanh()(self.fc5_o)

