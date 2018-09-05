import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Counter, act_fn, print_values


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

mask_size = Counter()


""" ****************** Modified (Michael Klachko) PNN Implementation ******************* """

class PerturbLayer(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, nmasks=None, level=None, filter_size=None,
                 debug=False, use_act=False, stride=1, act=None, unique_masks=False,
                 train_masks=False, noise_type='uniform', input_size=None):
        super(PerturbLayer, self).__init__()
        self.nmasks = nmasks    #per input channel
        self.unique_masks = unique_masks
        self.level = level
        self.filter_size = filter_size
        self.use_act = use_act
        self.act = act_fn(act)
        self.debug = debug
        self.train_masks = train_masks
        self.noise_type = noise_type

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
                self.act
            )
        else:
            if self.train_masks:
                noise_channels = in_channels if unique_masks else 1
                shape = (1, noise_channels, nmasks, input_size, input_size)
                self.noise = nn.Parameter(torch.Tensor(*shape), requires_grad=True).to(device)
                if noise_type == "uniform":
                    self.noise.data.uniform_(-level, level)
                elif self.noise_type == 'normal':
                    self.noise.data.normal_()
                    self.noise = self.noise * level
                else:
                    print('\n\nNoise type {} is not supported / understood\n\n'.format(self.noise_type))
                mask_size.update(self.noise.numel())
                print('Noise mask {:>20}  {:6.2f}k, accum. total: {:4.2f}M'.format(
                        str(list(self.noise.size())), self.noise.numel() / 1000., mask_size.get_total() / 1000000.))
            else:
                self.noise = nn.Parameter(torch.Tensor(0), requires_grad=False).to(device)

            self.noise = self.noise.cuda()

            if nmasks != 1:
                if out_channels % in_channels != 0:
                    print('\n\n\nnfilters must be divisible by 3 if using multiple noise masks per input channel\n\n\n')
                groups = in_channels
            else:
                groups = 1

            self.layers = nn.Sequential(
                #self.act,      #TODO orig code uses ReLU here
                #nn.BatchNorm2d(out_channels), #TODO: orig code uses BN here
                nn.Conv2d(in_channels*self.nmasks, out_channels, kernel_size=1, stride=1, groups=groups),
                nn.BatchNorm2d(out_channels),
                self.act
            )

    def forward(self, x):
        if self.filter_size > 0:
            return self.layers(x)  #image, conv, batchnorm, (relu?)
        else:
            bs, in_channels, h, v = list(x.size())
            if self.train_masks:
                #print(x.size())
                pass
            else:
                if self.noise.numel() == 0:
                    # if unique and nmasks=1: first layer shape (1, 3, nfilters, 32, 32) then (1, nfilters, nmasks, 32, 32)
                    # if unique and nmasks>1: first layer shape (1, 3, nmasks, 32, 32) then (1, nfilters, nmasks, 32, 32)

                    # if not unique and nmasks=1: first layer shape (1, 1, nfilters, 32, 32) then (1, 1, nmasks, 32, 32)
                    # if not unique and nmasks>1: first layer shape (1, 1, nmasks, 32, 32) then (1, 1, nmasks, 32, 32)
                    noise_channels = in_channels if self.unique_masks else 1

                    if self.noise_type == 'normal':
                        self.noise.resize_(1, noise_channels, self.nmasks, h, v).normal_()  #(1, 3, 128, 32, 32)
                        self.noise = self.noise * self.level
                    elif self.noise_type == 'uniform':
                        self.noise.resize_(1, noise_channels, self.nmasks, h, v).uniform_()  #(1, 3, 128, 32, 32)
                        self.noise = (2 * self.noise - 1) * self.level
                    else:
                        print('\n\nNoise type {} is not supported / understood\n\n'.format(self.noise_type))
                    mask_size.update(self.noise.numel())
                    print('Noise mask {:>20}  {:6.2f}k, accum. total: {:4.2f}M'.format(
                            str(list(self.noise.size())), self.noise.numel() / 1000., mask_size.get_total() / 1000000.))

            y = torch.add(x.unsqueeze(2), self.noise)  # (10, 3, 1, 32, 32) + (1, 3, 128, 32, 32) --> (10, 3, 128, 32, 32)

            if self.debug:
                print_values(x, self.noise, y, self.unique_masks)

            if self.use_act:
                y = self.act(y)

            y = y.view(bs, in_channels * self.nmasks, h, v)

            return self.layers(y)  #image, perturb, relu, conv1x1, batchnorm


class PerturbBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels=None, out_channels=None, stride=1, shortcut=None, nmasks=None, train_masks=False,
                 level=None, use_act=False, filter_size=None, act=None, unique_masks=False, noise_type=None, input_size=None):
        super(PerturbBasicBlock, self).__init__()
        self.shortcut = shortcut
        self.layers = nn.Sequential(
            PerturbLayer(in_channels=in_channels, out_channels=out_channels, nmasks=nmasks, input_size=input_size,
                         level=level, filter_size=filter_size, use_act=use_act, train_masks=train_masks,
                         act=act, unique_masks=unique_masks, noise_type=noise_type),  #perturb, relu, conv1x1
            nn.MaxPool2d(stride, stride),
            PerturbLayer(in_channels=out_channels, out_channels=out_channels, nmasks=nmasks, input_size=input_size//stride,
                         level=level, filter_size=filter_size, use_act=use_act, train_masks=train_masks,
                         act=act, unique_masks=unique_masks, noise_type=noise_type),  #perturb, relu, conv1x1
        )

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y


class PerturbResNet(nn.Module):
    def __init__(self, block, nblocks=None, avgpool=None, nfilters=None, nclasses=None, nmasks=None,
                 level=None, filter_size=None, first_filter_size=None, use_act=False, train_masks=False,
                 act=None, scale_noise=1, unique_masks=False, debug=False, noise_type=None):
        super(PerturbResNet, self).__init__()
        self.nfilters = nfilters
        self.unique_masks = unique_masks
        self.noise_type = noise_type
        self.train_masks = train_masks

        layers = [PerturbLayer(in_channels=3, out_channels=nfilters, nmasks=nmasks, level=level*scale_noise,
                debug=debug, filter_size=first_filter_size, use_act=use_act, train_masks=train_masks, input_size=32,
                act=act, unique_masks=self.unique_masks, noise_type=self.noise_type)]  # Perturb (+act?) OR conv, batchnorm, relu

        if first_filter_size == 7:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.pre_layers = nn.Sequential(*layers)
        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0], stride=1, level=level, nmasks=nmasks, use_act=True,
                                            filter_size=filter_size, act=act, input_size=32)
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2, level=level, nmasks=nmasks, use_act=True,
                                            filter_size=filter_size, act=act, input_size=32)
        self.layer3 = self._make_layer(block, 4*nfilters, nblocks[2], stride=2, level=level, nmasks=nmasks, use_act=True,
                                            filter_size=filter_size, act=act, input_size=16)
        self.layer4 = self._make_layer(block, 8*nfilters, nblocks[3], stride=2, level=level, nmasks=nmasks, use_act=True,
                                            filter_size=filter_size, act=act, input_size=8)
        self.avgpool = nn.AvgPool2d(avgpool, stride=1)
        self.linear = nn.Linear(8*nfilters*block.expansion, nclasses)

    def _make_layer(self, block, out_channels, nblocks, stride=1, level=0.2, nmasks=None, use_act=False,
                                            filter_size=None, act=None, input_size=None):
        shortcut = None
        if stride != 1 or self.nfilters != out_channels * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.nfilters, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.nfilters, out_channels, stride, shortcut, level=level, nmasks=nmasks, use_act=use_act,
                        filter_size=filter_size, act=act, unique_masks=self.unique_masks, noise_type=self.noise_type,
                        train_masks=self.train_masks, input_size=input_size))
        self.nfilters = out_channels * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.nfilters, out_channels, level=level, nmasks=nmasks, use_act=use_act, train_masks=self.train_masks,
                        filter_size=filter_size, act=act, unique_masks=self.unique_masks, noise_type=self.noise_type, input_size=input_size//stride))
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
    def __init__(self, nfilters=None, nclasses=None, nmasks=None, level=None, filter_size=None, linear=128,
                 debug=False, scale_noise=1, act='relu', use_act=False, first_filter_size=None,
                 dropout=None, unique_masks=False, train_masks=False, noise_type='uniform'):
        super(LeNet, self).__init__()
        if filter_size == 5:
            n = 5
        else:
            n = 4

        if nmasks == 1:
            first_nmasks = nfilters
        else:
            first_nmasks = nmasks

        self.linear1 = nn.Linear(nfilters*n*n, linear)
        self.linear2 = nn.Linear(linear, nclasses)
        self.dropout = nn.Dropout(p=dropout)
        self.act = act_fn(act)
        self.batch_norm = nn.BatchNorm1d(linear)

        self.first_layers = nn.Sequential(
            PerturbLayer(in_channels=1, out_channels=nfilters, nmasks=first_nmasks, level=level*scale_noise,
                         filter_size=first_filter_size, use_act=use_act,
                         act=act, unique_masks=unique_masks, train_masks=train_masks, noise_type=noise_type, input_size=28),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(2, 2, 0),
            #nn.AvgPool2d(2, 2, 0),
            PerturbLayer(in_channels=nfilters, out_channels=nfilters, nmasks=nmasks, level=level, filter_size=filter_size,
                         use_act=True, act=act, unique_masks=unique_masks,
                         debug=debug, train_masks=train_masks, noise_type=noise_type, input_size=14),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(2, 2, 0),
            #nn.AvgPool2d(2, 2, 0),
            PerturbLayer(in_channels=nfilters, out_channels=nfilters, nmasks=nmasks, level=level, filter_size=filter_size,
                         use_act=True, act=act, unique_masks=unique_masks,
                         train_masks=train_masks, noise_type=noise_type, input_size=7),
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
    def __init__(self, nfilters=None, nclasses=None, nmasks=None, level=None, filter_size=None,
                 linear=256, scale_noise=1, act='relu', use_act=False, first_filter_size=None,
                 dropout=None, unique_masks=False, debug=False, train_masks=False, noise_type='uniform'):
        super(CifarNet, self).__init__()
        if filter_size == 5:
            n = 5
        else:
            n = 4

        if nmasks == 1:
            first_nmasks = nfilters
        else:
            first_nmasks = nmasks

        self.in_channels = 1*nmasks if nmasks else nfilters
        self.linear1 = nn.Linear(nfilters*n*n, linear)
        self.linear2 = nn.Linear(linear, nclasses)
        self.dropout = nn.Dropout(p=dropout)
        self.act = act_fn(act)
        self.batch_norm = nn.BatchNorm1d(linear)

        print('\n\nscale_noise:', scale_noise, '\n\n')
        self.first_layers = nn.Sequential(
            PerturbLayer(in_channels=3, out_channels=nfilters, nmasks=first_nmasks, level=level*scale_noise,
                         unique_masks=unique_masks, filter_size=first_filter_size, use_act=use_act, input_size=32,
                         act=act, train_masks=train_masks, noise_type=noise_type),
            PerturbLayer(in_channels=nfilters, out_channels=nfilters, nmasks=nmasks, level=level, filter_size=filter_size,
                         debug=debug, use_act=True, act=act,
                         unique_masks=unique_masks, train_masks=train_masks, noise_type=noise_type, input_size=32),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(2, 2, 0),
            #nn.AvgPool2d(2, 2, 0),
            PerturbLayer(in_channels=nfilters, out_channels=nfilters, nmasks=nmasks, level=level, filter_size=filter_size,
                         use_act=True, act=act, unique_masks=unique_masks,
                         train_masks=train_masks, noise_type=noise_type, input_size=16),
            PerturbLayer(in_channels=nfilters, out_channels=nfilters, nmasks=nmasks, level=level, filter_size=filter_size,
                         use_act=True, act=act, unique_masks=unique_masks,
                         train_masks=train_masks, noise_type=noise_type, input_size=16),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            #nn.MaxPool2d(2, 2, 0),
            #nn.AvgPool2d(2, 2, 0),
            PerturbLayer(in_channels=nfilters, out_channels=nfilters, nmasks=nmasks, level=level, filter_size=filter_size,
                         use_act=True, act=act, unique_masks=unique_masks,
                         train_masks=train_masks, noise_type=noise_type, input_size=8),
            PerturbLayer(in_channels=nfilters, out_channels=nfilters, nmasks=nmasks, level=level, filter_size=filter_size,
                         use_act=True, act=act, unique_masks=unique_masks,
                         train_masks=train_masks, noise_type=noise_type, input_size=8),
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



"""************* Original PNN Implementation ****************"""

class NoiseLayer(nn.Module):
    def __init__(self, in_planes, out_planes, level):
        super(NoiseLayer, self).__init__()
        self.noise = nn.Parameter(torch.Tensor(0), requires_grad=False).to(device)
        self.level = level
        self.layers = nn.Sequential(
            nn.ReLU(True),
            nn.BatchNorm2d(in_planes),  #TODO paper does not use it!
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
        )

    def forward(self, x):
        if self.noise.numel() == 0:
            self.noise.resize_(x.data[0].shape).uniform_()   #fill with uniform noise
            self.noise = (2 * self.noise - 1) * self.level
            mask_size.update(self.noise.numel())
            print('Noise mask {:>20}  {:6.2f}k, accum. total: {:4.2f}M'.format(str(list(self.noise.size())),
                                    self.noise.numel() / 1000., mask_size.get_total() / 1000000.))
        y = torch.add(x, self.noise)
        return self.layers(y)   #input, perturb, relu, batchnorm, conv1x1


class NoiseBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut=None, level=0.2):
        super(NoiseBasicBlock, self).__init__()
        self.layers = nn.Sequential(
            NoiseLayer(in_planes, planes, level),  #perturb, relu, conv1x1
            nn.MaxPool2d(stride, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),  #TODO paper does not use it!
            NoiseLayer(planes, planes, level),  #perturb, relu, conv1x1
            nn.BatchNorm2d(planes),
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


class NoiseResNet(nn.Module):
    def __init__(self, block, nblocks, nfilters, nclasses, pool, level, first_filter_size=3):
        super(NoiseResNet, self).__init__()
        self.in_planes = nfilters
        if first_filter_size == 7:
            pool = 1
            self.pre_layers = nn.Sequential(
                nn.Conv2d(3, nfilters, kernel_size=first_filter_size, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(nfilters),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
            )
        elif first_filter_size == 3:
            pool = 4
            self.pre_layers = nn.Sequential(
                nn.Conv2d(3, nfilters, kernel_size=first_filter_size, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(nfilters),
                nn.ReLU(True),
            )
        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0], stride=1, level=level)
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2, level=level)
        self.layer3 = self._make_layer(block, 4*nfilters, nblocks[2], stride=2, level=level)
        self.layer4 = self._make_layer(block, 8*nfilters, nblocks[3], stride=2, level=level)
        self.avgpool = nn.AvgPool2d(pool, stride=1)
        self.linear = nn.Linear(8*nfilters*block.expansion, nclasses)

    def _make_layer(self, block, planes, nblocks, stride=1, level=0.2, filter_size=1):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, level=level))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, level=level))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.pre_layers(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.avgpool(x5)
        x7 = x6.view(x6.size(0), -1)
        x8 = self.linear(x7)
        return x8



""" *************** Reference ResNet Implementation (https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py) ****************** """

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, filter_size=3):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=filter_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=filter_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, nfilters=64, avgpool=4, nclasses=10, filter_size=None, first_filter_size=None):
        super(ResNet, self).__init__()
        self.in_planes = nfilters
        self.avgpool = avgpool

        self.conv1 = nn.Conv2d(3, nfilters, kernel_size=first_filter_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nfilters)
        self.layer1 = self._make_layer(block, nfilters, num_blocks[0], stride=1, filter_size=filter_size)
        self.layer2 = self._make_layer(block, nfilters*2, num_blocks[1], stride=2, filter_size=filter_size)
        self.layer3 = self._make_layer(block, nfilters*4, num_blocks[2], stride=2, filter_size=filter_size)
        self.layer4 = self._make_layer(block, nfilters*8, num_blocks[3], stride=2, filter_size=filter_size)
        self.linear = nn.Linear(nfilters*8*block.expansion, nclasses)

    def _make_layer(self, block, planes, num_blocks, stride, filter_size=3):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, filter_size=filter_size))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, self.avgpool)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(nfilters, avgpool=4, nclasses=10, nmasks=32, level=0.1, filter_size=0, first_filter_size=0,
             scale_noise=1, act='relu', use_act=True, dropout=0.5, unique_masks=False, noise_type='uniform', train_masks=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], nfilters=nfilters, avgpool=avgpool, nclasses=nclasses,
                      filter_size=filter_size, first_filter_size=first_filter_size)


def noiseresnet18(nfilters, avgpool=4, nclasses=10, nmasks=32, level=0.1, filter_size=0, first_filter_size=7,
                  scale_noise=1, act='relu', use_act=True, dropout=0.5, unique_masks=False, debug=False, noise_type='uniform', train_masks=False):
    return NoiseResNet(NoiseBasicBlock, [2, 2, 2, 2], nfilters=nfilters, pool=avgpool, nclasses=nclasses,
                       level=level, first_filter_size=first_filter_size)


def perturb_resnet18(nfilters, avgpool=4, nclasses=10, nmasks=32, level=0.1, filter_size=0, first_filter_size=0,
                     scale_noise=1, act='relu', use_act=True, dropout=0.5, unique_masks=False, debug=False, noise_type='uniform', train_masks=False):
    return PerturbResNet(PerturbBasicBlock, [2, 2, 2, 2], nfilters=nfilters, avgpool=avgpool, nclasses=nclasses,
                         scale_noise=scale_noise, nmasks=nmasks, level=level, filter_size=filter_size, train_masks=train_masks,
                         first_filter_size=first_filter_size, act=act, use_act=use_act, unique_masks=unique_masks,
                         debug=debug, noise_type=noise_type)


def lenet(nfilters, avgpool=None, nclasses=10, nmasks=32, level=0.1, filter_size=3, first_filter_size=0,
          scale_noise=1, act='relu', use_act=True, dropout=0.5, unique_masks=False, debug=False, noise_type='uniform', train_masks=False):
    return LeNet(nfilters=nfilters, nclasses=nclasses, nmasks=nmasks, level=level, filter_size=filter_size,
                 scale_noise=scale_noise, act=act, first_filter_size=first_filter_size,
                 use_act=use_act, dropout=dropout, unique_masks=unique_masks, debug=debug, noise_type=noise_type, train_masks=train_masks)


def cifarnet(nfilters, avgpool=None, nclasses=10, nmasks=32, level=0.1, filter_size=3, first_filter_size=0,
             scale_noise=1, act='relu', use_act=True, dropout=0.5, unique_masks=False, debug=False, noise_type='uniform', train_masks=False):
    return CifarNet(nfilters=nfilters, nclasses=nclasses, nmasks=nmasks, level=level, filter_size=filter_size,
                    scale_noise=scale_noise, act=act, use_act=use_act, first_filter_size=first_filter_size,
                    dropout=dropout, unique_masks=unique_masks, debug=debug, noise_type=noise_type, train_masks=train_masks)


