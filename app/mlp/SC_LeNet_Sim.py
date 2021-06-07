# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummaryX import summary

# %%
import os
cwd = os.getcwd()
print(cwd)
# model_path = cwd+"/saved_model_state_dict_bw_8test"
model_path = cwd+"/saved_model_state_dict_8"
print(model_path)

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# MNIST data loader
transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)

testset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=4)

# %%
class LeNet(nn.Module):

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

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32, 512)
        self.fc1_drop = nn.Dropout(0.6)
        self.fc2 = nn.Linear(512, 256)
        self.fc2_drop = nn.Dropout(0.6)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 32*32)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)        
        x = torch.clamp(x, -1, 1)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = torch.clamp(x, -1, 1)
        return F.log_softmax(self.fc3(x), dim=1)
    
model = Net()
model.cuda()


# %%
model.fc1.weight.data

# %%
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# %%
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size()[0]
        correct += (predicted == labels).sum().item()
    print(images.size())
print('Accuracy of the network on the 10000 test images: %f %%' % (
    100 * correct / total))

# %%
class BitStreamGen(object):
    def __init__(self, input, bitwidth=8, bipolar=True, dim=1, mode="Sobol"):
        super(BitStreamGen, self).__init__()
        self.input = input
        self.bitwidth = bitwidth
        self.bipolar = bipolar
        self.index = 0
        self.dim = dim
        self.mode = mode
        self.seq_len = pow(2,self.bitwidth)
        self.out = 0.0
        # random_sequence from sobol RNG
        if self.mode == "Sobol":
            self.random_sequence = torch.quasirandom.SobolEngine(self.dim).draw(self.seq_len).view(self.seq_len)
        elif self.mode == "Race":
            self.random_sequence = torch.tensor([x/self.seq_len for x in range(self.seq_len)])
        else:
            pass
        
        if self.bipolar is True:
            # convert to bipolar
            self.random_sequence.mul_(2).sub_(1)
    
    def Gen(self):
        self.out = torch.gt(self.input, self.random_sequence[self.index]).type(torch.float)
        self.index += 1
        return self.out
    

# %%
class ProgressivePrecision(object):
    def __init__(self, actual_value, bitwidth=8, bipolar=True, auto_print=False):
        super(ProgressivePrecision, self).__init__()
        self.actual_value = actual_value
        self.bitwidth = bitwidth
        self.bipolar = bipolar
        self.index = 0.0
        self.one_cnt = 0.0
        self.seq_len = pow(2,self.bitwidth)
        self.out_pp = 0.0
        self.error = 0.0
        self.auto_print = auto_print
        self.pp_list = []
    
    def Monitor(self, input):
        self.one_cnt += input
        self.index += 1
        self.out_pp = self.one_cnt / self.index
        if self.bipolar is True:
            self.out_pp = 2 * self.out_pp - 1
        self.pp_list.append(self.out_pp)
        
        self.err = self.out_pp - self.actual_value
        self.err_abs = self.err.abs()
        if self.auto_print is True:
            print("Progressive Error:", self.err)
        if self.index == self.seq_len:
            print("Final Error:", self.err)
            print("Final Value:", self.out_pp)

        return self.err
    
    def Stability(self, threshold):
        if self.index is self.seq_len:
            pass
        

# %%
import torch
class UnaryConv2d(torch.nn.modules.conv.Conv2d):
    """This is bipolar mul and non-scaled addition"""
    def __init__(self, in_channels, out_channels, kernel_size, output_shape,
                 binary_weight=torch.tensor([0]), binary_bias=torch.tensor([0]), bitwidth=8, 
                 stride=1, padding=0, dilation=1, 
                 groups=1, bias=True, padding_mode='zeros'):
        super(UnaryConv2d, self).__init__(in_channels, out_channels, kernel_size)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # data bit width
        self.buf_wght = binary_weight.clone().detach()
        if bias is True:
            self.buf_bias = binary_bias.clone().detach()
        self.bitwidth = bitwidth

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        self.groups = groups
        self.has_bias = bias
        self.padding_mode = padding_mode
        
        # random_sequence from sobol RNG
        self.rng = torch.quasirandom.SobolEngine(1).draw(pow(2,self.bitwidth)).view(pow(2,self.bitwidth))
        # convert to bipolar
        self.rng.mul_(2).sub_(1)
#         print(self.rng)

        # define the kernel linear
        self.kernel = torch.nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, 
                              stride=self.stride, padding=self.padding, dilation=self.dilation, 
                              groups=self.groups, bias=self.has_bias, padding_mode=self.padding_mode)

        # define the RNG index tensor for weight
        self.rng_wght_idx = torch.zeros(self.kernel.weight.size(), dtype=torch.long)
        self.rng_wght = self.rng[self.rng_wght_idx]
        assert (self.buf_wght.size() == self.rng_wght.size()
               ), "Input binary weight size of 'kernel' is different from true weight."
        
        # define the RNG index tensor for bias if available, only one is required for accumulation
        if self.has_bias is True:
            print("Has bias.")
            self.rng_bias_idx = torch.zeros(self.kernel.bias.size(), dtype=torch.long)
            self.rng_bias = self.rng[self.rng_bias_idx]
            assert (self.buf_bias.size() == self.rng_bias.size()
                   ), "Input binary bias size of 'kernel' is different from true bias."

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # inverse
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # define the kernel_inverse, no bias required
        self.kernel_inv = torch.nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, 
                              stride=self.stride, padding=self.padding, dilation=self.dilation, 
                              groups=self.groups, bias=False, padding_mode=self.padding_mode)
        
        # define the RNG index tensor for weight_inverse
        self.rng_wght_idx_inv = torch.zeros(self.kernel_inv.weight.size(), dtype=torch.long)
        self.rng_wght_inv = self.rng[self.rng_wght_idx_inv]
        assert (self.buf_wght.size() == self.rng_wght_inv.size()
               ), "Input binary weight size of 'kernel_inv' is different from true weight."
        
        self.in_accumulator = torch.zeros(output_shape)
        self.out_accumulator = torch.zeros(output_shape)
        self.output = torch.zeros(output_shape)
    
    def UnaryKernel_nonscaled_forward(self, input):
        # generate weight bits for current cycle
        self.rng_wght = self.rng[self.rng_wght_idx]
        self.kernel.weight.data = torch.gt(self.buf_wght, self.rng_wght).type(torch.float)
        print(self.rng_wght_idx.size())
        print(input.size())
        self.rng_wght_idx.add_(input.type(torch.long))
        if self.has_bias is True:
            self.rng_bias = self.rng[self.rng_bias_idx]
            self.kernel.bias.data = torch.gt(self.buf_bias, self.rng_bias).type(torch.float)
            self.rng_bias_idx.add_(1)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # inverse
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        self.rng_wght_inv = self.rng[self.rng_wght_idx_inv].type(torch.float)
        self.kernel_inv.weight.data = torch.le(self.buf_wght, self.rng_wght_inv).type(torch.float)
        self.rng_wght_idx_inv.add_(1).sub_(input.type(torch.long))
#         print(self.kernel(input).size())
        return self.kernel(input) + self.kernel_inv(1-input)
    
    def forward(self, input):
        self.in_accumulator.add_(self.UnaryKernel_nonscaled_forward(input))
#         .clamp_(-self.upper_bound, self.upper_bound)
        self.in_accumulator.sub_(self.offset)
        self.output = torch.gt(self.in_accumulator, self.out_accumulator).type(torch.float)
#         print("accumulator result:", self.in_accumulator, self.out_accumulator)
        self.out_accumulator.add_(self.output)
        return self.output
    

# %%
# conv = torch.nn.Conv2d(1,6,5)
# print(conv.weight.size())
# print(conv.bias.size())
# conv.weight.data = torch.rand(conv.weight.size()) * 2 -1
# conv.bias.data = torch.rand(conv.bias.size()) * 2 -1

# uconv = UnaryConv2d(1, 6, 5, (1, 6, 28, 28), conv.weight, conv.bias)

# inVec = torch.rand(1,1,32,32).floor()/256
# # print(inVec)
# outVec = conv(inVec)
# outVec.clamp_(-1.,1.)

# bsGen = BitStreamGen(inVec, mode="Sobol", bipolar=True)
# # bsGen = BitStreamGen(inVec,mode="Race")
# ipp = ProgressivePrecision(inVec, auto_print=False,bipolar=True)
# pp = ProgressivePrecision(outVec, auto_print=False,bipolar=True)

# with torch.no_grad():
#     for i in range(256):
#         input = bsGen.Gen()
#         output = uconv(input)
#         pp.Monitor(output)

# %%
# a(torch.zeros(1,1,32,32))

# %%
import torch
class UnaryLinear(torch.nn.modules.linear.Linear):
    def __init__(self, in_features, out_features, upper_bound,
                 binary_weight=torch.tensor([0]), binary_bias=torch.tensor([0]), bitwidth=8,
                 bias=True):
        super(UnaryLinear, self).__init__(in_features, out_features)
        
        self.in_features = in_features
        self.out_features = out_features
        self.upper_bound = upper_bound
        # bipolar accumulation
        self.offset = (in_features-1)/2
        
        # data bit width
        self.buf_wght = binary_weight.clone().detach()
        if bias is True:
            self.buf_bias = binary_bias.clone().detach()
        self.bitwidth = bitwidth
        
        self.has_bias = bias
        
        # random_sequence from sobol RNG
        self.rng = torch.quasirandom.SobolEngine(1).draw(pow(2,self.bitwidth)).view(pow(2,self.bitwidth))
        # convert to bipolar
        self.rng.mul_(2).sub_(1)
#         print(self.rng)

        # define the kernel linear
        self.kernel = torch.nn.Linear(self.in_features, self.out_features,
                                  bias=self.has_bias)

        # define the RNG index tensor for weight
        self.rng_wght_idx = torch.zeros(self.kernel.weight.size(), dtype=torch.long)
        self.rng_wght = self.rng[self.rng_wght_idx]
        assert (self.buf_wght.size() == self.rng_wght.size()
               ), "Input binary weight size of 'kernel' is different from true weight."
        
        # define the RNG index tensor for bias if available, only one is required for accumulation
        if self.has_bias is True:
            print("Has bias.")
            self.rng_bias_idx = torch.zeros(self.kernel.bias.size(), dtype=torch.long)
            self.rng_bias = self.rng[self.rng_bias_idx]
            assert (self.buf_bias.size() == self.rng_bias.size()
                   ), "Input binary bias size of 'kernel' is different from true bias."

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # inverse
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # define the kernel_inverse, no bias required
        self.kernel_inv = torch.nn.Linear(self.in_features, self.out_features,
                                  bias=False)
        
        # define the RNG index tensor for weight_inverse
        self.rng_wght_idx_inv = torch.zeros(self.kernel_inv.weight.size(), dtype=torch.long)
        self.rng_wght_inv = self.rng[self.rng_wght_idx_inv]
        assert (self.buf_wght.size() == self.rng_wght_inv.size()
               ), "Input binary weight size of 'kernel_inv' is different from true weight."
        
        self.in_accumulator = torch.zeros([1,out_features])
        self.out_accumulator = torch.zeros([1,out_features])
        self.output = torch.zeros([1,out_features])
#         self.cycle = 0

    def UnaryKernel_nonscaled_forward(self, input):
        # generate weight bits for current cycle
        self.rng_wght = self.rng[self.rng_wght_idx]
        self.kernel.weight.data = torch.gt(self.buf_wght, self.rng_wght).type(torch.float)
        self.rng_wght_idx.add_(input.type(torch.long))
        if self.has_bias is True:
            self.rng_bias = self.rng[self.rng_bias_idx]
            self.kernel.bias.data = torch.gt(self.buf_bias, self.rng_bias).type(torch.float)
            self.rng_bias_idx.add_(1)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # inverse
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        self.rng_wght_inv = self.rng[self.rng_wght_idx_inv].type(torch.float)
        self.kernel_inv.weight.data = torch.le(self.buf_wght, self.rng_wght_inv).type(torch.float)
        self.rng_wght_idx_inv.add_(1).sub_(input.type(torch.long))
        return self.kernel(input) + self.kernel_inv(1-input)

    def forward(self, input):
        self.in_accumulator.add_(self.UnaryKernel_nonscaled_forward(input))
#         .clamp_(-self.upper_bound, self.upper_bound)
        self.in_accumulator.sub_(self.offset)
        self.output = torch.gt(self.in_accumulator, self.out_accumulator).type(torch.float)
#         print("accumulator result:", self.in_accumulator, self.out_accumulator)
        self.out_accumulator.add_(self.output)
        return self.output
#         return self.UnaryKernel_nonscaled_forward(input)


# %%
# fc400 = nn.Linear(400, 120)
# fc400.weight = lenet.fc1.weight
# fc400.bias = lenet.fc1.bias
# ufc400 = UnaryLinear(400, 120, 256, lenet.fc1.weight, lenet.fc1.bias)
fc400 = nn.Linear(1024, 512, bias=True)
print(fc400.weight.size())
fc400.weight.data = model.fc1.weight
fc400.bias.data = model.fc1.bias
# fc400.weight.data = torch.rand(512,1024) * 2 -1
# fc400.bias.data = torch.rand(512) * 2 -1
print(fc400.weight.size())
# print(fc400.weight)


# ufc400 = UnaryLinear(400, 120, 256, fc400.weight, fc400.weight, bias=True)
ufc400 = UnaryLinear(1024, 512, 512, model.fc1.weight, model.fc1.bias, bias=True)


inVec = (((torch.rand(1024) * 2 - 1))*256).floor()/256
# print(inVec)
outVec = fc400(inVec)
outVec.clamp_(-1.,1.)

bsGen = BitStreamGen(inVec, mode="Sobol", bipolar=True)
# bsGen = BitStreamGen(inVec,mode="Race")
pp = ProgressivePrecision(outVec, auto_print=False,bipolar=True)
ipp = ProgressivePrecision(inVec, auto_print=False,bipolar=True)

with torch.no_grad():
    for i in range(256):
        input = bsGen.Gen()
#         print(ipp.Monitor(input))
        output = ufc400(input)
        pp.Monitor(output)



# %%


# %%
# input = bsGen.Gen()
# print(ipp.Monitor(input))
# output = ufc400(input)
# pp.Monitor(output)

# %%
# b.in_accumulator

# %%
import torch
import math
class UnaryScaledADD(torch.nn.modules.pooling.AvgPool2d):
    """unary scaled addition"""
    def __init__(self, kernel_size, input_shape, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True, divisor_override=None):
        super(UnaryScaledADD, self).__init__(kernel_size, input_shape)

        self.input_shape = input_shape
        
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        
        self.output_shape = list(input_shape)
        # data bit width
        if stride is None:
            if isinstance(kernel_size, int):
                self.scale = kernel_size*kernel_size
                self.output_shape[2] = int((input_shape[2] + 2 * padding - kernel_size) / kernel_size + 1)
                self.output_shape[3] = int((input_shape[3] + 2 * padding - kernel_size) / kernel_size + 1)
            elif isinstance(kernel_size, tuple):
                self.scale = kernel_size[0]*kernel_size[1]
                self.output_shape[2] = int((input_shape[2] + 2 * padding - kernel_size[0]) / stride[0] + 1)
                self.output_shape[3] = int((input_shape[3] + 2 * padding - kernel_size[1]) / stride[1] + 1)
        else:
            # to do
            pass
        
        self.in_accumulator = torch.zeros(self.output_shape)
        self.output = torch.zeros(self.output_shape)
        
        # define the kernel avgpool2d
        self.avgpool2d = torch.nn.AvgPool2d(self.kernel_size, 
                                            stride=self.stride, padding=self.padding, 
                                            ceil_mode=self.ceil_mode, 
                                            count_include_pad=self.count_include_pad, 
                                            divisor_override=self.divisor_override)
        
    def UnaryScaledADD_forward(self, input):
        self.in_accumulator.add_(self.avgpool2d(input))
        self.output = torch.ge(self.in_accumulator, 1).type(torch.float)
        self.in_accumulator.sub_(self.output)
        return self.output

    def forward(self, input):
        return self.UnaryScaledADD_forward(input)


# %%
# c = UnaryScaledADD(2, (1,1,28,28))

# %%
# c(torch.ones([1,1,28,28])/4)

# %%
import torch
import math
class UnaryCompare(torch.nn.modules.Module):
    """unary comparator"""
    def __init__(self, input_shape):
        super(UnaryCompare, self).__init__()

        self.input_shape = input_shape
        
        self.out_accumulator = torch.zeros([1,input_shape])
        self.out_acc_sign = torch.zeros([1,input_shape])
        self.output = torch.zeros([1,input_shape])

    def UnaryCompare_forward(self, input):
        self.out_acc_sign = torch.lt(self.out_accumulator, 0).type(torch.float)
        self.output = self.out_acc_sign + (1 - self.out_acc_sign) * input
        self.out_accumulator.add_(2 * self.output - 1)
        return self.output

    def forward(self, input):
        return self.UnaryCompare_forward(input)


# %%
class unaryNet(nn.Module):
    def __init__(self, model, prediction):
        super(unaryNet, self).__init__()
        self.model = model
        self.prediction = prediction
        self.fc1 = UnaryLinear(32*32, 512, 256, model.fc1.weight, model.fc1.bias)
        self.fc1_relu = UnaryCompare(512)
        self.fc2 = UnaryLinear(512, 256, 256, model.fc2.weight, model.fc2.bias)
        self.fc2_relu = UnaryCompare(256)
        self.fc3 = UnaryLinear(256, 10, 256, model.fc3.weight, model.fc3.bias)

    def forward(self, x):
        x = x.view(-1, 32*32)
        x = self.fc1_relu(self.fc1(x))
        x = self.fc2_relu(self.fc2(x))
        self.pp = ProgressivePrecision(self.prediction)
        return F.log_softmax(self.pp.Monitor(self.fc3(x)), dim=1)

# %%

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        uGEMMnet = unaryNet(model, outputs)
        bsGen = BitStreamGen(images)
        for i in range(256):
            uGEMM_out = uGEMMnet(bsGen.Gen())
            _, predicted = torch.max(uGEMM_out.data, 1)
            if predicted == labels:
                print("yes")
        total += labels.size()[0]
        correct += (predicted == labels).sum().item()
        print(predicted, labels)
print('Accuracy of the network on the 10000 test images: %f %%' % (
    100 * correct / total))

# %%
print('Accuracy of the network on the 10000 test images: %f %%' % (
    100 * correct / total))

# %%
