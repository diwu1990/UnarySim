import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
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
    print("{:30s}".format(name) + \
            ": min:" + "{:12f}".format(min) + \
            "; max:" + "{:12f}".format(max) + \
            "; outlier:" + "{:12f} %".format(outlier_ratio * 100))


def progerror_report(progerror, name="progerror", report_value=False, report_relative=False):
    if report_value:
        min = progerror.in_value.min().item()
        max = progerror.in_value.max().item()
        std, mean = torch.std_mean(progerror()[0])
        print("{:30s}".format(name) + \
                ", Binary   Value range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
                ", std," + "{:12f}".format(std) + \
                ", mean," + "{:12f}".format(mean))

        min = progerror()[0].min().item()
        max = progerror()[0].max().item()
        std, mean = torch.std_mean(progerror()[0])
        print("{:30s}".format(name) + \
                ", Unary    Value range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
                ", std," + "{:12f}".format(std) + \
                ", mean," + "{:12f}".format(mean))

    min = progerror()[1].min().item()
    max = progerror()[1].max().item()
    rmse = torch.sqrt(torch.mean(torch.square(progerror()[1])))
    std, mean = torch.std_mean(progerror()[1])
    print("{:30s}".format(name) + \
            ", Absolute Error range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
            ", std," + "{:12f}".format(std) + \
            ", mean," + "{:12f}".format(mean) + \
            ", rmse," + "{:12f}".format(rmse))

    if report_relative:
        relative_error = torch.nan_to_num(progerror()[1]/progerror()[0])
        min = relative_error.min().item()
        max = relative_error.max().item()
        rmse = torch.sqrt(torch.mean(torch.square(relative_error)))
        std, mean = torch.std_mean(relative_error)
        print("{:30s}".format(name) + \
                ", Relative Error range," + "{:12f}".format(min) + ", {:12f}".format(max) + \
                ", std," + "{:12f}".format(std) + \
                ", mean," + "{:12f}".format(mean) + \
                ", rmse," + "{:12f}".format(rmse))


class RoundingNoGrad(torch.autograd.Function):
    """
    RoundingNoGrad is a rounding operation which bypasses the input gradient to output directly.
    Original round()/floor()/ceil() opertions have a gradient of 0 everywhere, which is not useful 
    when doing approximate computing.
    This is something like the straight-through estimator (STE) for quantization-aware training.
    Code is taken from RAVEN (https://github.com/diwu1990/RAVEN/blob/master/pe/appr_utils.py)
    """
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)
    
    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input
    
    
class Round(torch.nn.Module):
    """
    Round is an operation to convert data to format (1, intwidth, fracwidth).
    """
    def __init__(self, intwidth=3, fracwidth=4) -> None:
        super(Round, self).__init__()
        self.intwidth = intwidth
        self.fracwidth = fracwidth
        self.max_val = (2**(intwidth + fracwidth) - 1)
        self.min_val = 0 - (2**(intwidth + fracwidth))

    def forward(self, input) -> Tensor:
        if input is None:
            return None
        else:
            return RoundingNoGrad.apply(input << self.fracwidth).clamp(self.min_val, self.max_val) >> self.fracwidth


def rshift_offset(input, weight, widthi, widthw, rounding="round", quantilei=1, quantilew=1):
    """
    This function calculate the right shift offset for the abs value of the input, weight and output.
    """
    with torch.no_grad():
        quantile_i_upper = 0.5 - quantilei / 2
        quantile_i_lower = 0.5 + quantilei / 2
        lower_bound_i = torch.quantile(input, quantile_i_lower)
        upper_bound_i = torch.quantile(input, quantile_i_upper)
        scale_i = torch.max(lower_bound_i.abs(), upper_bound_i.abs())
        imax_int = scale_i.log2()

        quantile_w_upper = 0.5 - quantilew / 2
        quantile_w_lower = 0.5 + quantilew / 2
        lower_bound_w = torch.quantile(weight, quantile_w_lower)
        upper_bound_w = torch.quantile(weight, quantile_w_upper)
        scale_w = torch.max(lower_bound_w.abs(), upper_bound_w.abs())
        wmax_int = scale_w.log2()

        if rounding == "round":
            imax_int = imax_int.round()
            wmax_int = wmax_int.round()
        elif rounding == "floor":
            imax_int = imax_int.floor()
            wmax_int = wmax_int.floor()
        elif rounding == "ceil":
            imax_int = imax_int.ceil()
            wmax_int = wmax_int.ceil()

        rshift_i = imax_int - widthi
        rshift_w = wmax_int - widthw
        rshift_o = max(widthi, widthw) - imax_int - wmax_int

        return rshift_i, rshift_w, rshift_o

