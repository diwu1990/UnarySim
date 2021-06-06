# %%
%load_ext autoreload
%autoreload 2

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

from UnarySim.kernel.utils import *

# %%
cwd = os.getcwd()
print(cwd)

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)

# %%
# MNIST data loader
transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root=cwd+'/../data/mnist', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.MNIST(root=cwd+'/../data/mnist', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, num_workers=4)

# %%
# model = LeNet()
# model = LeNet_clamp()
model = MLP3()
# model = MLP3_tanh()
# model = MLP3_clamp()
# model = MLP3_clamp_train()
model.to(device)
summary(model, torch.zeros((1, 1, 32, 32)).to(device))

# %%
bitwidth = 8
clipper = NN_SC_Weight_Clipper(bitwidth=bitwidth)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# %%
for epoch in range(20):  # loop over the dataset multiple times

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
    print(model.fc1.weight.max())
    print(model.fc1.weight.min())
    print(model.fc2.weight.max())
    print(model.fc2.weight.min())
    print(model.fc3.weight.max())
    print(model.fc3.weight.min())
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
model_path = cwd+"\saved_model_state_dict"+"_"+str(bitwidth)+"_no_clamp_new"
torch.save(model.state_dict(), model_path)

# %%
"""
# test load from state_dict
"""

# %%
model = MLP3_clamp_eval()
model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device)
# model.apply(clipper)
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %f %%' % (
    100 * correct / total))

# %%
print(model.fc1.weight.max())
print(model.fc1.weight.min())
print(model.fc1.bias.max())
print(model.fc1.bias.min())

print(model.fc2.weight.max())
print(model.fc2.weight.min())
print(model.fc2.bias.max())
print(model.fc2.bias.min())

print(model.fc3.weight.max())
print(model.fc3.weight.min())
print(model.fc3.bias.max())
print(model.fc3.bias.min())

# %%
# model.fc1.weight.data = model.fc1.weight*4
# model.fc1.bias.data = model.fc1.bias*4

# model.fc2.weight.data = model.fc2.weight*4
# model.fc2.bias.data = model.fc2.bias*4

# model.fc3.weight.data = model.fc3.weight*4
# model.fc3.bias.data = model.fc3.bias*4


# %%
# model_path = cwd+"\saved_model_state_dict"+"_"+str(bitwidth)+"_no_clamp_new"
# torch.save(model.state_dict(), model_path)

# %%
# model = MLP3_clamp_eval()
# model.load_state_dict(torch.load(model_path))
# model.eval()
# model.to(device)
# # model.apply(clipper)
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data[0].to(device), data[1].to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the 10000 test images: %f %%' % (
#     100 * correct / total))

# %%
# print(model.fc1.weight.max())
# print(model.fc1.weight.min())
# print(model.fc1.bias.max())
# print(model.fc1.bias.min())

# print(model.fc2.weight.max())
# print(model.fc2.weight.min())
# print(model.fc2.bias.max())
# print(model.fc2.bias.min())

# print(model.fc3.weight.max())
# print(model.fc3.weight.min())
# print(model.fc3.bias.max())
# print(model.fc3.bias.min())

# %%
