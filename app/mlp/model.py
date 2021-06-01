import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    