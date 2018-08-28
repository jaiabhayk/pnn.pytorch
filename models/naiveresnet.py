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

def act_fn(act):
    if act == 'relu':
        act_ = nn.ReLU(inplace=False)
    elif act == 'lrelu':
        act_ = nn.LeakyReLU(inplace=True)
    elif act == 'prelu':
        act_ = nn.PReLU()
    elif act == 'rrelu':
        act_ = nn.RReLU(inplace=True)
    elif act == 'elu':
        act_ = nn.ELU(inplace=True)
    elif act == 'selu':
        act_ = nn.SELU(inplace=True)
    elif act == 'tanh':
        act_ = nn.Tanh()
    elif act == 'sigmoid':
        act_ = nn.Sigmoid()
    return act_

class PerturbLayer(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, nmasks=None, level=None, filter_size=None, use_act=False, shape=None, stride=1, group=False, act=None):
        super(PerturbLayer, self).__init__()
        self.noise = nn.Parameter(torch.Tensor(0), requires_grad=False).to(device)
        #self.noise = nn.Parameter(torch.Tensor(*shape), requires_grad=False).to(device)
        #self.noise.data.uniform_(-level, level)
        self.noise = self.noise.cuda()
        #print('\n\n', nmasks, in_channels, '\n\n')

        self.nmasks = nmasks    #per input channel
        self.level = level
        self.filter_size = filter_size
        self.use_act = use_act
        self.act = act_fn(act)
        self.group = group

        print('act {}, use_act {}, level {}, nmasks {}, filter_size {}, group {}:'.format(self.act, self.use_act, self.level, self.nmasks, self.filter_size, self.group))

        if filter_size == 1:
            padding = 0
            bias = True
        elif filter_size == 3 or filter_size == 5:
            padding = 1
            bias = False
        elif filter_size == 7:
            stride = 2
            padding = 3
            bias = False

        if self.filter_size > 0:   #if filter_size=0, first_layer=[perturb, conv1x1] else first_layer=[convnxn], n=filter_size
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=filter_size, padding=padding, stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels),
                act_fn(act)
            )
        else:
            if self.group:
                groups = in_channels
            else:
                groups = 1
            self.layers = nn.Sequential(
                #self.act,      #TODO orig code uses ReLU here
                #nn.BatchNorm2d(out_channels), #TODO: orig code uses BN here
                nn.Conv2d(in_channels*self.nmasks, out_channels, kernel_size=1, stride=1, groups=groups),   #TODO try groups=in_channels
                nn.BatchNorm2d(out_channels),
                self.act
            )

    def forward(self, x):
        bs, in_channels, h, v = list(x.size())

        if self.filter_size > 0:
            return self.layers(x)  #image, conv, batchnorm, (relu?)
        else:
            if self.noise.numel() == 0:
                #self.noise.resize_(1, in_channels, self.nmasks, h, v).normal_()  #(1, 3, 128, 32, 32)
                #self.noise = self.noise * self.level
                self.noise.resize_(1, in_channels, self.nmasks, h, v).uniform_()  #(1, 3, 128, 32, 32)
                self.noise = (2 * self.noise - 1) * self.level
                mask_size.update(self.noise.numel())
                print('Noise mask {:>20}  {:6.2f}k, total: {:4.2f}M'.format(str(list(self.noise.size())), self.noise.numel() / 1000., mask_size.get_total() / 1000000.))

            y = torch.add(x.unsqueeze(2), self.noise)  # (10, 3, 1, 32, 32) + (1, 3, 128, 32, 32) --> (10, 3, 128, 32, 32)
            #print(list(x.unsqueeze(2).size()), list(self.noise.size()), list(y.size()))
            #np.set_printoptions(precision=5, linewidth=200, threshold=1000000, suppress=True)
            #print('\nx:', x.size(), x.data[0, 0, 0].cpu().numpy())
            #print('x:', x.size(), x.data[1, 0, 0].cpu().numpy())
            #print('noise:', self.noise.size(), self.noise.data[0, 0, 0, 0, :6].cpu().numpy())
            #print('\nx+noise:', y.size(), y.data[0, 0, 0].cpu().numpy())
            #print('x+noise:', y.size(), y.data[1, 0, 0].cpu().numpy())
            if self.use_act:
                y = self.act(y)
            y = y.view(bs, in_channels * self.nmasks, h, v)
            return self.layers(y)  #image, perturb, relu, conv1x1, batchnorm


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels=None, out_channels=None, stride=1, shortcut=None, nmasks=None, level=None, use_act=False,
                                            filter_size=None, shape=None, group=False, act=None):
        super(BasicBlock, self).__init__()
        self.layers = nn.Sequential(
            PerturbLayer(in_channels=in_channels, out_channels=out_channels, nmasks=nmasks, level=level, filter_size=filter_size, use_act=use_act,
                                        shape=(1, 1, nmasks, 28, 28), group=group, act=act),  #perturb, relu, conv1x1
            nn.MaxPool2d(stride, stride),
            PerturbLayer(in_channels=out_channels, out_channels=out_channels, nmasks=nmasks, level=level, filter_size=filter_size, use_act=use_act,
                                        shape=(1, 1, nmasks, 28, 28), group=group, act=act),  #perturb, relu, conv1x1
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
    def __init__(self, block, nblocks=None, avgpool=None, nfilters=None, nclasses=None, nmasks=None, level=None, filter_size=None,
                                            first_filter_size=None, use_act=False, shape=None, group=False, act=None, scale_noise=1):
        super(ResNet, self).__init__()
        self.nfilters = nfilters

        layers = [PerturbLayer(in_channels=3, out_channels=nfilters, nmasks=nmasks*nfilters, level=level*scale_noise,
                               filter_size=first_filter_size, use_act=use_act, shape=shape, group=group, act=act)]

        if first_filter_size == 7:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.pre_layers = nn.Sequential(*layers)
        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0], stride=1, level=level, nmasks=nmasks, use_act=use_act,
                                            filter_size=filter_size, shape=shape, group=group, act=act)
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2, level=level, nmasks=nmasks, use_act=use_act,
                                            filter_size=filter_size, shape=shape, group=group, act=act)
        self.layer3 = self._make_layer(block, 4*nfilters, nblocks[2], stride=2, level=level, nmasks=nmasks, use_act=use_act,
                                            filter_size=filter_size, shape=shape, group=group, act=act)
        self.layer4 = self._make_layer(block, 8*nfilters, nblocks[3], stride=2, level=level, nmasks=nmasks, use_act=use_act,
                                            filter_size=filter_size, shape=shape, group=group, act=act)
        self.avgpool = nn.AvgPool2d(avgpool, stride=1)
        self.linear = nn.Linear(8*nfilters*block.expansion, nclasses)

    def _make_layer(self, block, out_channels, nblocks, stride=1, level=0.2, nmasks=None, use_act=False,
                                            filter_size=None, shape=None, group=False, act=None):
        shortcut = None
        if stride != 1 or self.nfilters != out_channels * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.nfilters, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.nfilters, out_channels, stride, shortcut, level=level, nmasks=nmasks, use_act=use_act,
                                            filter_size=filter_size, shape=shape, group=group, act=act))
        self.nfilters = out_channels * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.nfilters, out_channels, level=level, nmasks=nmasks, use_act=use_act,
                                            filter_size=filter_size, shape=shape, group=group, act=act))
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
    def __init__(self, nfilters=None, nclasses=None, nmasks=None, level=None, filter_size=None, linear=128, group=False,
                                                    scale_noise=1, act='relu', use_act=False, first_filter_size=None, dropout=None):
        super(LeNet, self).__init__()
        if filter_size == 5:
            n = 5
        else:
            n = 4
        self.in_channels = 3*nmasks if nmasks else nfilters
        self.linear1 = nn.Linear(nfilters*n*n, linear)
        self.linear2 = nn.Linear(linear, nclasses)
        self.dropout = nn.Dropout(p=dropout)
        self.act = act_fn(act)
        self.batch_norm = nn.BatchNorm1d(linear)

        print('\n\nscale_noise:', scale_noise, '\n\n')
        self.first_layers = nn.Sequential(
            PerturbLayer(in_channels=1, out_channels=nfilters, nmasks=nmasks*nfilters, level=level*scale_noise, filter_size=first_filter_size, use_act=use_act,
                                        shape=(1, 1, nmasks, 28, 28), group=group, act=self.act),  #perturb, conv1x1
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(2, 2, 0),
            #nn.AvgPool2d(2, 2, 0),
            PerturbLayer(in_channels=nfilters, out_channels=nfilters, nmasks=nmasks, level=level, filter_size=filter_size, use_act=True,
                                        shape=(1, nfilters, nmasks, 14, 14), group=group, act=self.act),  #perturb, conv1x1
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(2, 2, 0),
            #nn.AvgPool2d(2, 2, 0),
            PerturbLayer(in_channels=nfilters, out_channels=nfilters, nmasks=nmasks, level=level, filter_size=filter_size, use_act=True,
                                        shape=(1, nfilters, nmasks, 7, 7), group=group, act=self.act),  #perturb, conv1x1
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            #nn.AvgPool2d(2, 2, 1),
        )

        self.last_layers = nn.Sequential(
            self.dropout,
            self.linear1,
            self.batch_norm,
            self.act,
            self.dropout,
            self.linear2,
        )

    def forward(self, x):
        x = self.first_layers(x)
        x = x.view(x.size(0), -1)
        x = self.last_layers(x)
        return x



class CifarNet(nn.Module):
    def __init__(self, nfilters=None, nclasses=None, nmasks=None, level=None, filter_size=None, linear=256, group=False,
                                                    scale_noise=1, act='relu', use_act=False, first_filter_size=None, dropout=None):
        super(CifarNet, self).__init__()
        if filter_size == 5:
            n = 5
        else:
            n = 4
        self.in_channels = 1*nmasks if nmasks else nfilters
        self.linear1 = nn.Linear(nfilters*n*n, linear)
        self.linear2 = nn.Linear(linear, nclasses)
        self.dropout = nn.Dropout(p=dropout)
        self.act = act_fn(act)
        self.batch_norm = nn.BatchNorm1d(linear)

        print('\n\nscale_noise:', scale_noise, '\n\n')
        self.first_layers = nn.Sequential(
            PerturbLayer(in_channels=3, out_channels=nfilters, nmasks=nmasks*nfilters, level=level*scale_noise, filter_size=first_filter_size, use_act=use_act,
                                        shape=(1, 1, nmasks, 28, 28), group=group, act=self.act),  #perturb, conv1x1
            PerturbLayer(in_channels=nfilters, out_channels=nfilters, nmasks=nmasks, level=level, filter_size=filter_size, use_act=True,
                                        shape=(1, nfilters, nmasks, 14, 14), group=group, act=self.act),  #perturb, conv1x1
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(2, 2, 0),
            #nn.AvgPool2d(2, 2, 0),
            PerturbLayer(in_channels=nfilters, out_channels=nfilters, nmasks=nmasks, level=level, filter_size=filter_size, use_act=True,
                                        shape=(1, nfilters, nmasks, 14, 14), group=group, act=self.act),  #perturb, conv1x1
            PerturbLayer(in_channels=nfilters, out_channels=nfilters, nmasks=nmasks, level=level, filter_size=filter_size, use_act=True,
                                        shape=(1, nfilters, nmasks, 14, 14), group=group, act=self.act),  #perturb, conv1x1
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(2, 2, 0),
            #nn.AvgPool2d(2, 2, 0),
            PerturbLayer(in_channels=nfilters, out_channels=nfilters, nmasks=nmasks, level=level, filter_size=filter_size, use_act=True,
                                        shape=(1, nfilters, nmasks, 7, 7), group=group, act=self.act),  #perturb, conv1x1
            PerturbLayer(in_channels=nfilters, out_channels=nfilters, nmasks=nmasks, level=level, filter_size=filter_size, use_act=True,
                                        shape=(1, nfilters, nmasks, 14, 14), group=group, act=self.act),  #perturb, conv1x1
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            #nn.AvgPool2d(2, 2, 1),
        )

        self.last_layers = nn.Sequential(
            self.dropout,
            self.linear1,
            self.batch_norm,
            self.act,
            self.dropout,
            self.linear2,
        )

    def forward(self, x):
        x = self.first_layers(x)
        #print(list(x.size()))
        x = x.view(x.size(0), -1)
        x = self.last_layers(x)
        return x



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
            nn.ReLU(True),  #TODO paper does not use it!
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


def resnet18(nfilters, avgpool=4, nclasses=10, nmasks=32, level=0.1, filter_size=0, first_filter_size=0, group=False, scale_noise=1, act='relu', use_act=True, dropout=0.5):
        return ResNet(BasicBlock, [2,2,2,2], nfilters=nfilters, avgpool=avgpool, nclasses=nclasses, group=group, scale_noise=scale_noise,
                       nmasks=nmasks, level=level, filter_size=filter_size, first_filter_size=first_filter_size, act=act, use_act=use_act)

def lenet(nfilters, avgpool=None, nclasses=10, nmasks=32, level=0.1, filter_size=3, first_filter_size=0, group=False, scale_noise=1, act='relu', use_act=True, dropout=0.5):
    return LeNet(nfilters=nfilters, nclasses=nclasses, nmasks=nmasks, level=level, filter_size=filter_size, group=group, scale_noise=scale_noise,
                    act=act, first_filter_size=first_filter_size, use_act=use_act, dropout=dropout)

def cifarnet(nfilters, avgpool=None, nclasses=10, nmasks=32, level=0.1, filter_size=3, first_filter_size=0, group=False, scale_noise=1, act='relu', use_act=True, dropout=0.5):
    return CifarNet(nfilters=nfilters, nclasses=nclasses, nmasks=nmasks, level=level, filter_size=filter_size, group=group, scale_noise=scale_noise,
                    act=act, use_act=use_act, first_filter_size=first_filter_size, dropout=dropout)

def resnet34(nfilters, level=0.1):
    return ResNet(NoiseBasicBlock, [3,4,6,3], nfilters=nfilters, level=level)

def resnet50(nfilters, level=0.1):
    return ResNet(NoiseBottleneck, [3,4,6,3], nfilters=nfilters, level=level)

def resnet101(nfilters, level=0.1):
    return ResNet(NoiseBottleneck, [3,4,23,3], nfilters=nfilters, level=level)

def resnet152(nfilters, level=0.1):
    return ResNet(NoiseBottleneck, [3,8,36,3], nfilters=nfilters, level=level)
