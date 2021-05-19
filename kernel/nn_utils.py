import torch
import torch.nn as nn
import torch.nn.functional as F
from UnarySim.kernel.linear import UnaryLinear
from UnarySim.kernel.relu import UnaryReLU
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


class LeNet(nn.Module):
    """
    This is a standard LeNet
    """
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv1(x)
        x = F.avg_pool2d(x, (2, 2))
        x = F.relu(x)
        # If the size is a square you can only specify a single number
        x = self.conv2(x)
        x = F.avg_pool2d(x, (2, 2))
        x = F.relu(x)
        
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class LeNet_clamp(nn.Module):
    """
    This is a standard LeNet, but with weights/biases clamp to (-1, 1)
    """
    def __init__(self):
        super(LeNet_clamp, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.conv1(x)
        x = torch.clamp(x, -1, 1)
        x = F.avg_pool2d(x, (2, 2))
        x = F.relu(x)
        # If the size is a square you can only specify a single number
        x = self.conv2(x)
        x = torch.clamp(x, -1, 1)
        x = F.avg_pool2d(x, (2, 2))
        x = F.relu(x)
        
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = torch.clamp(x, -1, 1)
        x = F.relu(self.fc2(x))
        x = torch.clamp(x, -1, 1)
        x = self.fc3(x)
        return x
    
    
class MLP3(nn.Module):
    def __init__(self, width=512, p=0.5):
        super(MLP3, self).__init__()
        self.fc1 = nn.Linear(32*32, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, 10)
        
        self.fc1_out = torch.zeros(1)
        self.do1 = nn.Dropout(p=p)
        self.relu1_out = torch.zeros(1)
        self.fc2_out = torch.zeros(1)
        self.do2 = nn.Dropout(p=p)
        self.relu2_out = torch.zeros(1)
        self.fc3_out = torch.zeros(1)

    def forward(self, x):
        x = x.view(-1, 32*32)
        self.fc1_out = self.fc1(x)
        self.relu1_out = F.relu(self.do1(self.fc1_out))
        self.fc2_out = self.fc2(self.relu1_out)
        self.relu2_out = F.relu(self.do2(self.fc2_out))
        self.fc3_out = self.fc3(self.relu2_out)
        return F.softmax(self.fc3_out, dim=1)
    
    
class MLP3_clamp_eval(nn.Module):
    def __init__(self):
        super(MLP3_clamp_eval, self).__init__()
        self.fc1 = nn.Linear(32*32, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        
        self.fc1_out = torch.zeros(1)
        self.relu1_out = torch.zeros(1)
        self.fc2_out = torch.zeros(1)
        self.relu2_out = torch.zeros(1)
        self.fc3_out = torch.zeros(1)

    def forward(self, x):
        x = x.view(-1, 32*32)
        self.fc1_out = self.fc1(x).clamp(-1, 1)
        self.relu1_out = F.relu(self.fc1_out)
        self.fc2_out = self.fc2(self.relu1_out).clamp(-1, 1)
        self.relu2_out = F.relu(self.fc2_out)
        self.fc3_out = self.fc3(self.relu2_out).clamp(-1, 1)
        return F.softmax(self.fc3_out, dim=1)

    
class MLP3_clamp_train(nn.Module):
    """
    For unary training, activation clamp is better to be after relu.
    no difference for inference whether clamp is after or before relu.
    """
    def __init__(self):
        super(MLP3_clamp_train, self).__init__()
        self.fc1 = nn.Linear(32*32, 512)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 32*32)
        x = F.relu(self.fc1(x)).clamp(-1, 1)
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x)).clamp(-1, 1)
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), dim=1)
    
    

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
    
    h = (h_w[0] - 1)*stride[0] - sum(pad[0]) + dialation[0]*(kernel_size[0]-1) + out_pad[0] + 1
    w = (h_w[1] - 1)*stride[1] - sum(pad[1]) + dialation[1]*(kernel_size[1]-1) + out_pad[1] + 1
    
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
        
    p_h = -(h_w_out[0] - 1 - out_pad[0] - dialation[0]*(kernel_size[0]-1) - (h_w[0] - 1)*stride[0]) / 2
    p_w = -(h_w_out[1] - 1 - out_pad[1] - dialation[1]*(kernel_size[1]-1) - (h_w[1] - 1)*stride[1]) / 2
    
    return (math.floor(p_h/2), math.ceil(p_h/2)), (math.floor(p_w/2), math.ceil(p_w/2))