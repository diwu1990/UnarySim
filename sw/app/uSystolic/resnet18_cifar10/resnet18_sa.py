'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from UnarySim.sw.kernel.conv import UnaryConv2dSA
from UnarySim.sw.kernel.linear import UnaryLinearSA

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, cycle=None):
        super(BasicBlock, self).__init__()
        self.conv1 = UnaryConv2dSA(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, cycle=cycle)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = UnaryConv2dSA(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False, cycle=cycle)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                UnaryConv2dSA(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, cycle=cycle),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, cycle=None):
        super(Bottleneck, self).__init__()
        self.conv1 = UnaryConv2dSA(in_planes, planes, kernel_size=1, bias=False, cycle=cycle)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = UnaryConv2dSA(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False, cycle=cycle)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = UnaryConv2dSA(planes, self.expansion *
                               planes, kernel_size=1, bias=False, cycle=cycle)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                UnaryConv2dSA(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, cycle=cycle),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cycle=None):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = UnaryConv2dSA(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False, cycle=cycle)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, cycle=cycle)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, cycle=cycle)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, cycle=cycle)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, cycle=cycle)
        self.linear = UnaryLinearSA(512*block.expansion, num_classes, cycle=cycle)

    def _make_layer(self, block, planes, num_blocks, stride, cycle):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, cycle))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(cycle=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], cycle=cycle)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()