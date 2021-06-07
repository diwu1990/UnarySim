# %%
%load_ext autoreload
%autoreload 2

# %%
import torch
from UnarySim.kernel.linear_st import LinearST
from UnarySim.sw.stream.gen import RNG, SourceGen, BSGen
from UnarySim.sw.metric.metric import ProgError
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummaryX import summary
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from UnarySim.kernel.utils import *

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
bitwidth = 8
# layer_width = 512
# lr = 0.001

# layer_width = 1024
# lr = 0.001

layer_width = 2048
lr = 0.0001

# layer_width = 4096
# lr = 0.0001

# layer_width = 16384
# lr = 0.0001

cwd = os.getcwd()
print(cwd)
model_path = cwd+"\saved_model_state_dict"+"_"+str(bitwidth)+"_bitwidth_"+str(layer_width)+"_layerwidth_"+str(lr)+"_lr"

# %%
# MNIST data loader
transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
datadir = 'd:/project/Anaconda3/Lib/site-packages/UnarySim/sw/test/mlp/data/mnist'
trainset = torchvision.datasets.MNIST(root=datadir, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.MNIST(root=datadir, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, num_workers=4)

model = MLP3(layer_width)
model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device)
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size()[0]
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %f %%' % (
    100 * correct / total))

# %%
# print(images.max(), images.min())
# print("model fc1 wght: max:", model.fc1.weight.max().item(), "min:", model.fc1.weight.min().item())
# print("model fc1 bias: max:", model.fc1.bias.max().item(),   "min:", model.fc1.bias.min().item())
# print("model fc2 wght: max:", model.fc2.weight.max().item(), "min:", model.fc2.weight.min().item())
# print("model fc2 bias: max:", model.fc2.bias.max().item(),   "min:", model.fc2.bias.min().item())
# print("model fc3 wght: max:", model.fc3.weight.max().item(), "min:", model.fc3.weight.min().item())
# print("model fc3 bias: max:", model.fc3.bias.max().item(),   "min:", model.fc3.bias.min().item())

# %%
# fc1_wght_freq = np.fft.fft2(model.fc1.weight.clone().detach().cpu().numpy())
# print(fc1_wght_freq)
# print(images.shape)
# print(images[0][0].shape)
# image_freq = np.fft.fft2(images[0][0].clone().detach().cpu().numpy() * 255)
# print(image_freq)
# print(images[0][0].clone().detach().cpu().numpy().max())
# im_array = np.asarray(images[0][0].clone().detach().cpu().numpy() * 255)
# plt.imshow(im_array, cmap='gray', vmin=0, vmax=255)
# plt.show()

# %%
rng = "Sobol"
rng_width = 8
bias = True
population = 1
rng_stride = 3

correct = 0
total = 0
total_cnt = 100

fc1 = LinearST(32*32, layer_width, model.fc1.weight/model.fc1.weight.abs().max().item(), model.fc1.bias/model.fc1.weight.abs().max().item(), bias=bias, 
               mode="bipolar", rng=rng, rng_width=rng_width, rng_stride=rng_stride, population=population).to(device)
fc2 = LinearST(layer_width,   layer_width, model.fc2.weight/model.fc2.weight.abs().max().item(), model.fc2.bias/model.fc2.weight.abs().max().item(), bias=bias, 
               mode="bipolar", rng=rng, rng_width=rng_width, rng_stride=rng_stride, population=population).to(device)
fc3 = LinearST(layer_width,   10,  model.fc3.weight/model.fc3.weight.abs().max().item(), model.fc3.bias/model.fc3.weight.abs().max().item(), bias=bias, 
               mode="bipolar", rng=rng, rng_width=rng_width, rng_stride=rng_stride, population=population).to(device)
with torch.no_grad():
    index = 0
    for data in testloader:
        index += 1
        if index > total_cnt:
            break
        images, labels = data[0].to(device), data[1].to(device)
        x = images.view(-1, 32*32)
        fc1_out = fc1(x)
#         print(fc1_out.max().item(), fc1_out.min().item())
        fc1_scale = fc1_out.abs().max()
        fc1_out = fc1_out / fc1_scale
#         print(fc1_out.max().item(), fc1_out.min().item())
        fc1_act = F.relu(fc1_out)

        fc2_out = fc2(fc1_act)
        fc2_scale = fc2_out.abs().max()
        fc2_out = fc2_out / fc2_scale
        fc2_act = F.relu(fc2_out)

        fc3_out = fc3(fc2_act)
        
        outputs = model(images)
        fc1_ref_out = model.fc1_out
        fc2_ref_out = model.fc2_out
        fc3_ref_out = model.fc3_out
        fc1_ref_act = model.relu1_out
        fc2_ref_act = model.relu2_out
#         print(fc1_ref_out)
#         print(fc1_out)
#         print(fc1_ref_out - fc1_out)
#         print(fc2_ref_act)
#         print(fc2_act)
#         print(torch.sum(torch.gt(fc2_ref_act, 0).type(torch.float)))
#         print(torch.sum(torch.gt(fc2_act, 0).type(torch.float)))
        
#         print(torch.max(outputs.data, 1)[1])
#         print(torch.max(fc3_out.data, 1)[1])
        
        _, predicted = torch.max(fc3_out.data, 1)
        total += labels.size()[0]
        correct += (predicted == labels).sum().item()
        
print('Accuracy of the network on the 10000 test images: %f %%' % (
    100 * correct / total))

# %%
model_fp = MLP3(layer_width)
bitwidth = 2
model_fp.eval()
model_fp.to(device)
model_fp.fc1.weight.data = model.fc1.weight.mul(2**bitwidth).round().div(2**bitwidth).clone().detach()
model_fp.fc1.bias.data   = model.fc1.bias.mul(2**bitwidth).round().div(2**bitwidth).clone().detach()
model_fp.fc2.weight.data = model.fc2.weight.mul(2**bitwidth).round().div(2**bitwidth).clone().detach()
model_fp.fc2.bias.data   = model.fc2.bias.mul(2**bitwidth).round().div(2**bitwidth).clone().detach()
model_fp.fc3.weight.data = model.fc3.weight.mul(2**bitwidth).round().div(2**bitwidth).clone().detach()
model_fp.fc3.bias.data   = model.fc3.bias.mul(2**bitwidth).round().div(2**bitwidth).clone().detach()

correct = 0
total = 0
total_cnt = 100

with torch.no_grad():
    index = 0
    for data in testloader:
        index += 1
        if index > total_cnt:
            break
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model_fp(images.mul(2**bitwidth).round().div(2**bitwidth))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size()[0]
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %f %%' % (
    100 * correct / total))

# %%
"""
512

population : fp mul acc : new design acc : xnornet acc : fp model acc : 

         1 :        13% :            13% :         24% :              : 
         2 :        11% :            14% :         24% :           8% : bw0
         4 :        53% :            17% :         24% :          29% : bw1
         8 :        87% :            21% :         24% :          98% : bw2
        16 :        91% :            48% :         24% :          99% : bw3
        32 :        97% :            46% :         24% :          99% : bw4
        64 :        96% :            54% :         24% :          99% : bw5
       128 :        97% :            80% :         24% :          99% : bw6
       256 :        97% :            97% :         24% :          99% : bw7

"""

# %%
"""
1024

population : fp mul acc : new design acc : xnornet acc : fp model acc : 

         1 :         8% :             8% :         27% :              : 
         2 :        18% :            16% :         27% :           8% : bw0
         4 :        88% :            18% :         27% :          93% : bw1
         8 :        96% :            14% :         27% :          98% : bw2
        16 :        98% :            70% :         27% :          98% : bw3
        32 :        98% :            70% :         27% :          98% : bw4
        64 :        98% :            98% :         27% :          98% : bw5
       128 :        98% :            98% :         27% :          98% : bw6
       256 :        98% :            98% :         27% :          98% : bw7

"""

# %%
"""
2048

population : fp mul acc : new design acc : xnornet acc : fp model acc : 

         1 :         9% :             9% :         48% :              : 
         2 :        14% :            15% :         48% :           8% : bw0
         4 :        84% :            28% :         48% :           8% : bw1
         8 :        96% :            32% :         48% :           8% : bw2
        16 :        98% :            71% :         48% :          24% : bw3
        32 :        98% :            86% :         48% :          96% : bw4
        64 :        98% :            98% :         48% :          98% : bw5
       128 :        98% :            98% :         48% :          98% : bw6
       256 :        98% :            98% :         48% :          98% : bw7

"""

# %%
