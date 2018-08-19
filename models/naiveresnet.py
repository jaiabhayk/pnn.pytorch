# naiveresnet.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class FirstLayer(nn.Module):
    def __init__(self, in_channels, out_channels, nmasks, level, conv_first, perturb_first):
        super(FirstLayer, self).__init__()
        self.noise = nn.Parameter(torch.Tensor(0), requires_grad=False).to(device)
        self.nmasks = nmasks    #per input channel
        self.level = level
        self.perturb_first = perturb_first
        self.conv_first = conv_first
        if conv_first == 1:
            stride = 1
            padding = 0
            bias = True
        elif conv_first == 3:
            stride = 1
            padding = 1
            bias = False
        elif conv_first == 7:
            stride = 2
            padding = 3
            bias = False

        if self.conv_first > 0:
            self.conv_first_layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=conv_first, padding=padding, stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),  #TODO: not sure if it's needed
            )
        elif self.perturb_first:
            self.perturb_first_layers = nn.Sequential(
                nn.ReLU(True),
                #nn.BatchNorm2d(out_channels), #TODO: orig code uses BN here
                nn.Conv2d(in_channels*nmasks, out_channels, kernel_size=1, stride=1, groups=1),   #TODO try groups=3
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        bs, in_channels, h, v = list(x.size())

        if self.conv_first > 0:
            return self.conv_first_layers(x)  #image, conv, batchnorm, (relu?)

        elif self.perturb_first == "broadcast":
            if self.noise.numel() == 0:
                self.noise.resize_(1, in_channels, self.nmasks, h, v).uniform_()  #(1, 3, 9, 32, 32)
                self.noise = (2 * self.noise - 1) * self.level
            y = torch.add(x.unsqueeze(2), self.noise)  # (10, 3, 1, 32, 32) + (1, 3, 9, 32, 32) --> (10, 3, 9, 32, 32)
            return self.perturb_first_layers(y.view(bs, in_channels * self.nmasks, h, v))  #image, perturb, relu, conv1x1, batchnorm

        elif self.perturb_first != "broadcast":
            raise NotImplementedError('{} perturbation method has not been implemented.'.format(self.perturb_first))


class NoiseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, level):
        super(NoiseLayer, self).__init__()
        self.noise = nn.Parameter(torch.Tensor(0), requires_grad=False).to(device)
        self.level = level
        self.layers = nn.Sequential(
            nn.ReLU(True),
            nn.BatchNorm2d(in_channels),  #NOTE paper does not use it!
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            #nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        if self.noise.numel() == 0:
            self.noise.resize_(x.data[0].shape).uniform_()   #fill with uniform noise
            self.noise = (2 * self.noise - 1) * self.level
        y = torch.add(x, self.noise)
        return self.layers(y)   #input, perturb, relu, conv1x1, batchnorm

class NoiseBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None, level=0.2):
        super(NoiseBasicBlock, self).__init__()
        self.layers = nn.Sequential(
            NoiseLayer(in_channels, out_channels, level),  #perturb, relu, conv1x1
            nn.MaxPool2d(stride, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),  #NOTE paper does not use it!
            NoiseLayer(out_channels, out_channels, level),  #perturb, relu, conv1x1
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, shortcut=None, level=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut:
            residual = self.shortcut(x)
        out += residual
        out = F.relu(out)
        return out

class NoiseBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None, level=0.2):
        super(NoiseBottleneck, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            NoiseLayer(out_channels, out_channels, level),
            nn.MaxPool2d(stride, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y

class ResNet(nn.Module):
    def __init__(self, block=None, nblocks=None, avgpool=None, nfilters=None, nclasses=None,
                            nmasks=None, level=None, conv_first=None, perturb_first=None):
        super(ResNet, self).__init__()
        self.in_channels = 3 * nmasks if nmasks else nfilters
        layers = [FirstLayer(in_channels=3, out_channels=self.in_channels, nmasks=nmasks,
                                  level=level, conv_first=conv_first, perturb_first=perturb_first)]

        if conv_first == 7:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.pre_layers = nn.Sequential(*layers)
        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0], stride=1, level=level)
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2, level=level)
        self.layer3 = self._make_layer(block, 4*nfilters, nblocks[2], stride=2, level=level)
        self.layer4 = self._make_layer(block, 8*nfilters, nblocks[3], stride=2, level=level)
        self.avgpool = nn.AvgPool2d(avgpool, stride=1)
        self.linear = nn.Linear(8*nfilters*block.expansion, nclasses)

    def _make_layer(self, block, out_channels, nblocks, stride=1, level=0.2):
        shortcut = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, shortcut, level=level))
        self.in_channels = out_channels * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_channels, out_channels, level=level))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def resnet18(nfilters, avgpool=4, nclasses=10, nmasks=32, level=0.1, conv_first=3, perturb_first=None, perturb=None):
    if perturb:
        return ResNet(NoiseBasicBlock, [2,2,2,2], nfilters=nfilters, avgpool=avgpool, nclasses=nclasses,
                       nmasks=nmasks, level=level, conv_first=conv_first, perturb_first=perturb_first)
    else:
        return ResNet(BasicBlock, [2, 2, 2, 2], nfilters=nfilters, avgpool=avgpool, nclasses=nclasses, conv_first=conv_first)

def resnet34(nfilters, level=0.1):
    return ResNet(NoiseBasicBlock, [3,4,6,3], nfilters=nfilters, level=level)

def resnet50(nfilters, level=0.1):
    return ResNet(NoiseBottleneck, [3,4,6,3], nfilters=nfilters, level=level)

def resnet101(nfilters, level=0.1):
    return ResNet(NoiseBottleneck, [3,4,23,3], nfilters=nfilters, level=level)

def resnet152(nfilters, level=0.1):
    return ResNet(NoiseBottleneck, [3,8,36,3], nfilters=nfilters, level=level)
