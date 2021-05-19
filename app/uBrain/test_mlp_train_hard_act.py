# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummaryX import summary
import matplotlib.pyplot as plt
import time
import os

from UnarySim.sw.kernel.nn_utils import *

# %%
cwd = os.getcwd()
cwd = "D:/project/Anaconda3/Lib/site-packages/UnarySim/sw/app/mlp/"
print(cwd)

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %%
# MNIST data loader
transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root=cwd+'/data/mnist', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.MNIST(root=cwd+'/data/mnist', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, num_workers=4)

# %%
layer_width = 512
total_epoch = 40
lr = 0.001
print(total_epoch)

# %%
class ScaleReLU1(nn.Hardtanh):
    """
    clip the input when it is larger than 1.
    """
    def __init__(self, inplace: bool = False):
        super(ScaleReLU1, self).__init__(0., 1., inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

# %%
class ScaleHardsigmoid(nn.Module):
    """
    valid input range is (-1, +1).
    """
    def __init__(self, scale=3):
        super(ScaleHardsigmoid, self).__init__()
        self.scale = scale

    def forward(self, x) -> str:
        return nn.Hardsigmoid()(x * self.scale)

# %%
class MLP3_hardsig(nn.Module):
    def __init__(self, width=512, p=0.5):
        super(MLP3_hardsig, self).__init__()
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
#         self.relu1_out = ScaleHardsigmoid()(self.do1(self.fc1_out))
        self.relu1_out = nn.Sigmoid()(self.do1(self.fc1_out))
#         self.relu1_out = nn.Hardtanh()(self.do1(self.fc1_out))
#         self.relu1_out = nn.Tanh()(self.do1(self.fc1_out))
#         self.relu1_out = F.relu6(self.do1(self.fc1_out))
#         self.relu1_out = ScaleReLU1()(self.do1(self.fc1_out))
#         self.relu1_out = F.relu(self.do1(self.fc1_out))
        self.fc2_out = self.fc2(self.relu1_out)
#         self.relu2_out = ScaleHardsigmoid()(self.do2(self.fc2_out))
        self.relu2_out = nn.Sigmoid()(self.do2(self.fc2_out))
#         self.relu2_out = nn.Hardtanh()(self.do2(self.fc2_out))
#         self.relu2_out = nn.Tanh()(self.do2(self.fc2_out))
#         self.relu2_out = F.relu6(self.do2(self.fc2_out))
#         self.relu2_out = ScaleReLU1()(self.do2(self.fc2_out))
#         self.relu2_out = F.relu(self.do2(self.fc2_out))
        self.fc3_out = self.fc3(self.relu2_out)
        return F.softmax(self.fc3_out, dim=1)

# %%
model = MLP3_hardsig(layer_width)
model.to(device)
summary(model, torch.zeros((1, 1, 32, 32)).to(device))

# %%
bitwidth = 8
clipper = NN_SC_Weight_Clipper(bitwidth=bitwidth)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# %%
for epoch in range(total_epoch):  # loop over the dataset multiple times
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
    
    model.eval()
#     model.apply(clipper)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Train - Epoch %d, Loss: %f, Test Accuracy: %f %%' \
          % (epoch, loss.detach().cpu().item(), 100 * correct / total))

print('Finished Training')

# %%
"""
ScaleHardsigmoid: 95

sigmoid: 94

Hardtanh: 97

Tanh: 97

relu6: 97

ScaleReLU1: 97

relu: 97
"""

# %%
