import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NN_SC_Weight_Clipper(object):
    """
    This is a clipper for weights and bias of neural networks
    """
    def __init__(self, frequency=1, mode="bipolar", method="clip", bitwidth=8):
        self.frequency = frequency
        # "unipolar" or "bipolar"
        self.mode = mode
        # "clip" or "norm"
        self.method = method
        self.scale = 2 ** bitwidth

    def __call__(self, module):
        # filter the variables to get the ones you want
        if self.frequency > 1:
            self.method = "clip"
        else:
            self.method = "norm"
                
        if hasattr(module, 'weight'):
            w = module.weight.data
            self.clipping(w)
        
        if hasattr(module, 'bias'):
            w = module.bias.data
            self.clipping(w)
        
        self.frequency = self.frequency + 1
            
    def clipping(self, w):
        if self.mode == "unipolar":
            if self.method == "norm":
                w.sub_(torch.min(w)).div_(torch.max(w) - torch.min(w)) \
                .mul_(self.scale).round_().clamp_(0.0,self.scale).div_(self.scale)
            elif self.method == "clip":
                w.clamp_(0.0,1.0) \
                .mul_(self.scale).round_().clamp_(0.0,self.scale).div_(self.scale)
            else:
                raise TypeError("unknown method type '{}' in SC_Weight, should be 'clip' or 'norm'"
                                .format(self.method))
        elif self.mode == "bipolar":
            if self.method == "norm":
                w.sub_(torch.min(w)).div_(torch.max(w) - torch.min(w)).mul_(2).sub_(1) \
                .mul_(self.scale/2).round_().clamp_(-self.scale/2,self.scale/2).div_(self.scale/2)
            elif self.method == "clip":
                w.clamp_(-1.0,1.0) \
                .mul_(self.scale/2).round_().clamp_(-self.scale/2,self.scale/2).div_(self.scale/2)
            else:
                raise TypeError("unknown method type '{}' in SC_Weight, should be 'clip' or 'norm'"
                                .format(self.method))
        else:
            raise TypeError("unknown mode type '{}' in SC_Weight, should be 'unipolar' or 'bipolar'"
                            .format(self.mode))


def num2tuple(num):
    return num if isinstance(num, tuple) else (num, num)


def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    h_w, kernel_size, stride, pad, dilation = num2tuple(h_w), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(pad), num2tuple(dilation)
    pad = num2tuple(pad[0]), num2tuple(pad[1])
    
    h = math.floor((h_w[0] + sum(pad[0]) - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
    w = math.floor((h_w[1] + sum(pad[1]) - dilation[1]*(kernel_size[1]-1) - 1) / stride[1] + 1)
    
    return h, w


def convtransp2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1, out_pad=0):
    h_w, kernel_size, stride, pad, dilation, out_pad = num2tuple(h_w), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(pad), num2tuple(dilation), num2tuple(out_pad)
    pad = num2tuple(pad[0]), num2tuple(pad[1])
    
    h = (h_w[0] - 1)*stride[0] - sum(pad[0]) + dilation[0]*(kernel_size[0]-1) + out_pad[0] + 1
    w = (h_w[1] - 1)*stride[1] - sum(pad[1]) + dilation[1]*(kernel_size[1]-1) + out_pad[1] + 1
    
    return h, w


def conv2d_get_padding(h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1):
    h_w_in, h_w_out, kernel_size, stride, dilation = num2tuple(h_w_in), num2tuple(h_w_out), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(dilation)
    
    p_h = ((h_w_out[0] - 1)*stride[0] - h_w_in[0] + dilation[0]*(kernel_size[0]-1) + 1)
    p_w = ((h_w_out[1] - 1)*stride[1] - h_w_in[1] + dilation[1]*(kernel_size[1]-1) + 1)
    
    return (math.floor(p_h/2), math.ceil(p_h/2)), (math.floor(p_w/2), math.ceil(p_w/2))


def convtransp2d_get_padding(h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1, out_pad=0):
    h_w_in, h_w_out, kernel_size, stride, dilation, out_pad = num2tuple(h_w_in), num2tuple(h_w_out), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(dilation), num2tuple(out_pad)
        
    p_h = -(h_w_out[0] - 1 - out_pad[0] - dilation[0]*(kernel_size[0]-1) - (h_w_in[0] - 1)*stride[0]) / 2
    p_w = -(h_w_out[1] - 1 - out_pad[1] - dilation[1]*(kernel_size[1]-1) - (h_w_in[1] - 1)*stride[1]) / 2
    
    return (math.floor(p_h/2), math.ceil(p_h/2)), (math.floor(p_w/2), math.ceil(p_w/2))


def truncated_normal(t, mean=0.0, std=0.01):
    torch.nn.init.normal_(t, mean=mean, std=std)
    while True:
        cond = torch.logical_or(t < mean - 2*std, t > mean + 2*std)
        if torch.sum(cond):
            t = torch.where(cond, torch.nn.init.normal_(torch.ones_like(t), mean=mean, std=std), t)
        else:
            break
    return t


def tensor_unary_outlier(tensor, name="tensor"):
    min = tensor.min().item()
    max = tensor.max().item()
    outlier = torch.sum(torch.gt(tensor, 1)) + torch.sum(torch.lt(tensor, -1))
    outlier_ratio = outlier / torch.prod(torch.tensor(tensor.size()))
    print("{:20s}".format(name) + \
            ": min:" + "{:10f}".format(min) + \
            "; max:" + "{:10f}".format(max) + \
            "; outlier:" + "{:10f} %".format(outlier_ratio * 100))
