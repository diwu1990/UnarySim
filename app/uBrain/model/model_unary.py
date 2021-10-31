import math
import warnings
import numbers
from typing import List, Tuple, Optional, overload, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from UnarySim.kernel.conv import FSUConv2d
from UnarySim.kernel.linear import FSULinear
from UnarySim.kernel.relu import FSUReLU
from UnarySim.kernel.rnn import FSUMGUCell
from UnarySim.metric.metric import SourceGen, RNG, BSGen, ProgError
from UnarySim.kernel.utils import progerror_report


class Cascade_CNN_RNN(torch.nn.Module):
    """
    This is the unary version of the cascade CNN RNN for BCI
    """
    def __init__(self, 
                    input_sz=(10, 11), 
                    linear_act="hardtanh", 
                    cnn_chn=32, 
                    cnn_kn_sz=3, 
                    cnn_padding=0, 
                    fc_sz=1024, 
                    rnn="mgu", 
                    rnn_win_sz=10, 
                    rnn_hidden_sz=1024, 
                    rnn_hard=True, 
                    bias=True, 
                    init_std=None, 
                    keep_prob=0.5, 
                    num_class=5, 
                    bitwidth=8, 
                    acc_depth=12, 
                    relu_sr_depth=5, 
                    rng="Sobol", 
                    conv1_weight=None, 
                    conv1_bias=None, 
                    conv2_weight=None, 
                    conv2_bias=None, 
                    fc3_weight=None, 
                    fc3_bias=None, 
                    rnncell4_weight_ih=None, 
                    rnncell4_bias_ih=None, 
                    rnncell4_weight_hh=None, 
                    rnncell4_bias_hh=None, 
                    fc5_weight=None, 
                    fc5_bias=None, 
                    debug=False):
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
        self.bitwidth = bitwidth
        self.acc_depth = acc_depth
        self.relu_sr_depth = relu_sr_depth
        self.rng = rng
        self.conv1_weight = conv1_weight
        self.conv1_bias = conv1_bias
        self.conv2_weight = conv2_weight
        self.conv2_bias = conv2_bias
        self.fc3_weight = fc3_weight
        self.fc3_bias = fc3_bias
        self.rnncell4_weight_ih = rnncell4_weight_ih
        self.rnncell4_bias_ih = rnncell4_bias_ih
        self.rnncell4_weight_hh = rnncell4_weight_hh
        self.rnncell4_bias_hh = rnncell4_bias_hh
        self.fc5_weight = fc5_weight
        self.fc5_bias = fc5_bias
        self.debug = debug

    def build_model(self, device):
        # CNN
        self.conv1 = FSUConv2d(1, self.cnn_chn , self.cnn_kn_sz, bias=self.bias, padding=self.cnn_padding, 
                                binary_weight=self.conv1_weight, binary_bias=self.conv1_bias, bitwidth=self.bitwidth, scaled=False, depth=self.acc_depth).to(device)
        self.conv1_act = FSUReLU(depth=self.relu_sr_depth).to(device)
        self.conv2 = FSUConv2d(self.cnn_chn, self.cnn_chn*2, self.cnn_kn_sz, bias=self.bias, padding=self.cnn_padding, 
                                binary_weight=self.conv2_weight, binary_bias=self.conv2_bias, bitwidth=self.bitwidth, scaled=False, depth=self.acc_depth).to(device)
        self.conv2_act = FSUReLU(depth=self.relu_sr_depth).to(device)
        self.fc3 = FSULinear((self.input_sz[0]+2*2*(self.cnn_padding-1))*(self.input_sz[1]+2*2*(self.cnn_padding-1))*self.cnn_chn*2, self.fc_sz, bias=self.bias, 
                                binary_weight=self.fc3_weight, binary_bias=self.fc3_bias, bitwidth=self.bitwidth, scaled=False, depth=self.acc_depth).to(device)
        self.fc3_act = FSUReLU(depth=self.relu_sr_depth).to(device)

        # RNN
        self.rnncell4 = FSUMGUCell(self.fc_sz, self.rnn_hidden_sz, bias=self.bias, 
                                    binary_weight_ih=self.rnncell4_weight_ih, binary_bias_ih=self.rnncell4_bias_ih, 
                                    binary_weight_hh=self.rnncell4_weight_hh, binary_bias_hh=self.rnncell4_bias_hh, 
                                    bitwidth=self.bitwidth, depth=self.acc_depth).to(device)

        # MLP
        self.fc5 = FSULinear(self.rnn_hidden_sz, self.num_class, bias=self.bias, 
                                binary_weight=self.fc5_weight, binary_bias=self.fc5_bias, bitwidth=self.bitwidth, scaled=False, depth=self.acc_depth).to(device)

    def forward(self, input, binary_fm_dict=None):
        # input is (batch, win, h, w)
        # binary_fm_dict will be {"conv1_act_o", "conv2_act_o", "fc3_trans_o", 
        #                           "gate_i", 
        #                           "gate_h", 
        #                           "forgetgate", 
        #                           "newgate_prod", 
        #                           "newgate", 
        #                           "forgetgate_inv_prod", 
        #                           "forgetgate_prod", 
        #                           "rnn_out", 
        #                           "logits"}
        
        hx = torch.zeros(input.size()[0], self.rnn_hidden_sz, dtype=input.dtype, device=input.device)
        oPEM = ProgError(binary_fm_dict["logits"]).to(input.device)

        for win in range(self.rnn_win_sz):
            self.build_model(input.device)
            # for win-th window, (:, win, :, :) is picked to generate iSRC with shape (batch, 1, h, w)
            iSRC = SourceGen(input[:, win, :, :].unsqueeze(1), bitwidth=self.bitwidth)().to(input.device)
            iRNG = RNG(self.bitwidth, 1, self.rng)().to(input.device)
            iBSG = BSGen(iSRC, iRNG).to(input.device)

            # hx is (batch, hidden_sz)
            hSRC = SourceGen(hx, bitwidth=self.bitwidth)().to(input.device)
            hRNG = RNG(self.bitwidth, 1, self.rng)().to(input.device)
            hBSG = BSGen(hSRC, hRNG).to(input.device)
            
            hPEM = ProgError(torch.zeros(input.size()[0], self.rnn_hidden_sz, dtype=input.dtype, device=input.device)).to(input.device)

            # conv1_act_o in binary model has the shape of (batch*win, chn, h, w), resize to (batch, win, chn, h, w)
            conv1_act_o_PEM = ProgError(binary_fm_dict["conv1_act_o"].view(-1, 
                                                                            self.rnn_win_sz, 
                                                                            binary_fm_dict["conv1_act_o"].size(1), 
                                                                            binary_fm_dict["conv1_act_o"].size(2), 
                                                                            binary_fm_dict["conv1_act_o"].size(3))[:, win, :, :, :]).to(input.device)
            # conv2_act_o in binary model has the shape of (batch*win, chn, h, w), resize to (batch, win, chn, h, w)
            conv2_act_o_PEM = ProgError(binary_fm_dict["conv2_act_o"].view(-1, 
                                                                            self.rnn_win_sz, 
                                                                            binary_fm_dict["conv2_act_o"].size(1), 
                                                                            binary_fm_dict["conv2_act_o"].size(2), 
                                                                            binary_fm_dict["conv2_act_o"].size(3))[:, win, :, :, :]).to(input.device)
            # fc3_trans_o_PEM in binary model has the shape of (win, batch, fc_sz)
            fc3_trans_o_PEM = ProgError(binary_fm_dict["fc3_trans_o"][win]).to(input.device)

            gate_i_PEM = ProgError(binary_fm_dict["gate_i"][win]).to(input.device)
            gate_h_PEM = ProgError(binary_fm_dict["gate_h"][win]).to(input.device)
            forgetgate_PEM = ProgError(binary_fm_dict["forgetgate"][win]).to(input.device)
            newgate_prod_PEM = ProgError(binary_fm_dict["newgate_prod"][win]).to(input.device)
            newgate_PEM = ProgError(binary_fm_dict["newgate"][win]).to(input.device)
            forgetgate_inv_prod_PEM = ProgError(binary_fm_dict["forgetgate_inv_prod"][win]).to(input.device)
            forgetgate_prod_PEM = ProgError(binary_fm_dict["forgetgate_prod"][win]).to(input.device)

            # rnn_out in binary model has the shape of (win, batch, hidden_sz)
            rnn_out_PEM = ProgError(binary_fm_dict["rnn_out"][win]).to(input.device)

            for i in range(2**self.bitwidth):
                # CNN
                idx                 = torch.zeros_like(iSRC).type(torch.long)
                self.conv1_i        = iBSG(idx + i)
                self.conv1_o        = self.conv1(self.conv1_i)
                self.conv1_act_o    = self.conv1_act(self.conv1_o)
                conv1_act_o_PEM.Monitor(self.conv1_act_o)

                self.conv2_o        = self.conv2(self.conv1_act_o)
                self.conv2_act_o    = self.conv2_act(self.conv2_o)
                conv2_act_o_PEM.Monitor(self.conv2_act_o)
                
                self.fc3_i          = self.conv2_act_o.view(self.conv2_act_o.shape[0], -1)
                self.fc3_o          = self.fc3(self.fc3_i)
                self.fc3_act_o      = self.fc3_act(self.fc3_o)
                self.fc3_view_o     = self.fc3_act_o
                self.fc3_trans_o    = self.fc3_view_o
                fc3_trans_o_PEM.Monitor(self.fc3_trans_o)

                # RNN
                hdx                 = torch.zeros_like(hSRC).type(torch.long)
                self.hx_i           = hBSG(hdx + i)
                self.hx_o           = self.rnncell4(self.fc3_trans_o, self.hx_i)
                gate_i_PEM.Monitor(self.rnncell4.gate_i)
                gate_h_PEM.Monitor(self.rnncell4.gate_h)
                forgetgate_PEM.Monitor(self.rnncell4.forgetgate)
                newgate_prod_PEM.Monitor(self.rnncell4.newgate_prod)
                newgate_PEM.Monitor(self.rnncell4.newgate)
                forgetgate_inv_prod_PEM.Monitor(self.rnncell4.forgetgate_inv_prod)
                forgetgate_prod_PEM.Monitor(self.rnncell4.forgetgate_prod)
                hPEM.Monitor(self.hx_o)
                rnn_out_PEM.Monitor(self.hx_o)
                # MLP
                if win == self.rnn_win_sz - 1:
                    self.fc5_i      = self.hx_o
                    self.fc5_o      = self.fc5(self.fc5_i)
                    oPEM.Monitor(self.fc5_o)
            hx = hPEM()[0]
            progerror_report(conv1_act_o_PEM, "conv1_act_o_PEM")
            progerror_report(conv2_act_o_PEM, "conv2_act_o_PEM")
            progerror_report(fc3_trans_o_PEM, "fc3_trans_o_PEM")
            progerror_report(gate_i_PEM, "gate_i_PEM")
            progerror_report(gate_h_PEM, "gate_h_PEM")
            progerror_report(forgetgate_PEM, "forgetgate_PEM")
            progerror_report(newgate_prod_PEM, "newgate_prod_PEM")
            progerror_report(newgate_PEM, "newgate_PEM")
            progerror_report(forgetgate_inv_prod_PEM, "forgetgate_inv_prod_PEM")
            progerror_report(forgetgate_prod_PEM, "forgetgate_prod_PEM")
            progerror_report(rnn_out_PEM, "rnn_out_PEM")
            print()
        return oPEM()[0]

