# model.py

import math
from torch import nn
import models
import losses
import torch.nn.init as init

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

class Model:

    def __init__(self, args):
        self.cuda = args.cuda
        self.perturb = args.perturb
        self.nfilters = args.nfilters
        self.nclasses = args.nclasses
        self.nchannels = args.nchannels
        self.nblocks = args.nblocks
        self.nlayers = args.nlayers
        self.level = args.level
        self.net_type = args.net_type
        self.avgpool = args.avgpool
        self.nmasks = args.nmasks
        self.first_conv = args.first_conv

        if args.dataset_train.startswith("CIFAR"):
            self.nclasses = 10
            if self.first_conv < 7:
                self.avgpool = 4
            elif self.first_conv == 7:
                self.avgpool = 1
        elif args.dataset_train.startswith("ImageNet"):
            self.nclasses = 1000
            if self.first_conv < 7:
                self.avgpool = 14  #TODO
            elif self.first_conv == 7:
                self.avgpool = 7

    def setup(self, checkpoints):

        model = getattr(models, self.net_type)(
            nfilters=self.nfilters,
            avgpool=self.avgpool,
            nclasses=self.nclasses,
            nmasks=self.nmasks,
            level=self.level,
            first_conv=self.first_conv,
            perturb=self.perturb
        )

        loss_fn = losses.Classification()

        if checkpoints.latest('resume') == None:
            model.apply(weights_init)
        else:
            tmp = checkpoints.load(checkpoints.latest('resume'))
            model.load_state_dict(tmp)

        if self.cuda:
            model = model.cuda()
            loss_fn = loss_fn.cuda()

        return model, loss_fn