# naiveresnet.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Counter

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

mask_size = Counter()

class FirstLayer(nn.Module):
    def __init__(self, in_channels, out_channels, nmasks, level, first_conv, relu=False, shape=None):
        super(FirstLayer, self).__init__()
        self.noise = nn.Parameter(torch.Tensor(0), requires_grad=False).to(device)

        #self.noise = nn.Parameter(torch.Tensor(*shape), requires_grad=False).to(device)
        #self.noise.data.uniform_(-level, level)

        self.nmasks = nmasks    #per input channel
        self.level = level
        self.first_conv = first_conv
        self.relu = relu

        if first_conv == 1:
            stride = 1
            padding = 0
            bias = True
        elif first_conv == 3 or first_conv == 5:
            stride = 1
            padding = 1
            bias = False
        elif first_conv == 7:
            stride = 2
            padding = 3
            bias = False

        if self.first_conv > 0:   #if first_conv=0, first_layer=[perturb, conv1x1] else first_layer=[convnxn], n=first_conv
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=first_conv, padding=padding, stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),  #TODO: not sure if it's needed
            )
        else:
            self.layers = nn.Sequential(
                #nn.ReLU(True),      #TODO orig code uses ReLU here
                #nn.BatchNorm2d(out_channels), #TODO: orig code uses BN here
                nn.Conv2d(in_channels*nmasks, out_channels, kernel_size=1, stride=1, groups=1),   #TODO try groups=in_channels
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        bs, in_channels, h, v = list(x.size())

        if self.first_conv > 0:
            return self.layers(x)  #image, conv, batchnorm, (relu?)
        else:
            if self.noise.numel() == 0:
                self.noise.resize_(1, in_channels, self.nmasks, h, v).uniform_()  #(1, 3, 9, 32, 32)
                self.noise = (2 * self.noise - 1) * self.level
                mask_size.update(self.noise.numel())
                print('Noise masks:\n{:>18}  {:6.2f}k, total: {:4.2f}M'.format(str(list(self.noise.size())), self.noise.numel() / 1000., mask_size.get_total() / 1000000.))
            #print(list(x.unsqueeze(2).size()), list(self.noise.size()))
            y = torch.add(x.unsqueeze(2), self.noise)  # (10, 3, 1, 32, 32) + (1, 3, 9, 32, 32) --> (10, 3, 9, 32, 32)
            #np.set_printoptions(precision=5, linewidth=200, threshold=1000000, suppress=True)
            #print('\nx:', x.size(), x.data[0, 0, 0].cpu().numpy())
            #print('x:', x.size(), x.data[1, 0, 0].cpu().numpy())
            #print('noise:', self.noise.size(), self.noise.data[0, 0, 0, 0, :6].cpu().numpy())
            #print('\nx+noise:', y.size(), y.data[0, 0, 0].cpu().numpy())
            #print('x+noise:', y.size(), y.data[1, 0, 0].cpu().numpy())
            if self.relu:
                y = F.relu(y)
            return self.layers(y.view(bs, in_channels * self.nmasks, h, v))  #image, perturb, relu, conv1x1, batchnorm


class NoiseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, level):
        super(NoiseLayer, self).__init__()
        self.noise = nn.Parameter(torch.Tensor(0), requires_grad=False).to(device)
        self.level = level
        self.layers = nn.Sequential(
            nn.ReLU(True),
            #nn.BatchNorm2d(in_channels),  #NOTE paper does not use it!
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            #nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        if self.noise.numel() == 0:
            self.noise.resize_(x.data[0].shape).uniform_()   #fill with uniform noise
            self.noise = (2 * self.noise - 1) * self.level
            mask_size.update(self.noise.numel())
            print('{:>18}  {:6.2f}k, total: {:4.2f}M'.format(str(list(self.noise.size())), self.noise.numel() / 1000., mask_size.get_total() / 1000000.))
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
                            nmasks=None, level=None, first_conv=None):
        super(ResNet, self).__init__()
        self.in_channels = 3 * nmasks if nmasks else nfilters
        layers = [FirstLayer(in_channels=3, out_channels=self.in_channels, nmasks=nmasks,
                                  level=level, first_conv=first_conv)]

        if first_conv == 7:
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

class LeNet(nn.Module):
    def __init__(self, nfilters=None, nclasses=None, nmasks=None, level=None, first_conv=None, linear=128):
        super(LeNet, self).__init__()
        if first_conv == 5:
            n = 5
        else:
            n = 2
        self.in_channels = 1*nmasks if nmasks else nfilters
        self.linear1 = nn.Linear(nfilters*n*n, linear)
        self.linear2 = nn.Linear(linear, nclasses)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(True)

        self.layers = nn.Sequential(
            FirstLayer(in_channels=1, out_channels=nfilters, nmasks=nmasks, level=level, first_conv=first_conv, shape=(1, 1, nmasks, 28, 28)),  #perturb, conv1x1
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(True),
            #NoiseLayer(self.in_channels, nfilters, level)
            FirstLayer(in_channels=nfilters, out_channels=nfilters, nmasks=nmasks, level=level, first_conv=first_conv, relu=True, shape=(1, nfilters, nmasks, 14, 14)),  #perturb, conv1x1
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(True),
            FirstLayer(in_channels=nfilters, out_channels=nfilters, nmasks=nmasks, level=level, first_conv=first_conv, relu=True, shape=(1, nfilters, nmasks, 7, 7)),  #perturb, conv1x1
            nn.MaxPool2d(kernel_size=3, stride=2), nn.ReLU(True),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x - self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


def resnet18(nfilters, avgpool=4, nclasses=10, nmasks=32, level=0.1, first_conv=3, perturb=None):
    if perturb:
        return ResNet(NoiseBasicBlock, [2,2,2,2], nfilters=nfilters, avgpool=avgpool, nclasses=nclasses,
                       nmasks=nmasks, level=level, first_conv=first_conv)
    else:
        return ResNet(BasicBlock, [2, 2, 2, 2], nfilters=nfilters, avgpool=avgpool, nclasses=nclasses, first_conv=first_conv)

def lenet(nfilters, avgpool=None, nclasses=10, nmasks=32, level=0.1, first_conv=3, perturb=None):
    return LeNet(nfilters=nfilters, nclasses=nclasses, nmasks=nmasks, level=level, first_conv=first_conv)


def resnet34(nfilters, level=0.1):
    return ResNet(NoiseBasicBlock, [3,4,6,3], nfilters=nfilters, level=level)

def resnet50(nfilters, level=0.1):
    return ResNet(NoiseBottleneck, [3,4,6,3], nfilters=nfilters, level=level)

def resnet101(nfilters, level=0.1):
    return ResNet(NoiseBottleneck, [3,4,23,3], nfilters=nfilters, level=level)

def resnet152(nfilters, level=0.1):
    return ResNet(NoiseBottleneck, [3,8,36,3], nfilters=nfilters, level=level)
