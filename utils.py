# utils.py

import os
import copy
import numpy as np
import torch
import torch.nn as nn

def readtextfile(filename):
    with open(filename) as f:
        content = f.readlines()
    f.close()
    return content

def writetextfile(data, filename):
    with open(filename, 'w') as f:
        f.writelines(data)
    f.close()

def delete_file(filename):
    if os.path.isfile(filename) == True:
        os.remove(filename)

def eformat(f, prec, exp_digits):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%+0*d"%(mantissa, exp_digits+1, int(exp))

def saveargs(args):
    path = args.logs
    if os.path.isdir(path) == False:
        os.makedirs(path)
    with open(os.path.join(path,'args.txt'), 'w') as f:
        for arg in vars(args):
            f.write(arg+' '+str(getattr(args,arg))+'\n')

def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1)
            nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=1e-3)
            if m.bias:
                nn.init.constant(m.bias, 0)


class Checkpoints:
    def __init__(self, args):
        self.dir_save = args.save
        self.dir_load = args.resume

        if os.path.isdir(self.dir_save) == False:
            os.makedirs(self.dir_save)

    def latest(self, name):
        if name == 'resume':
            if self.dir_load == None:
                return None
            else:
                return self.dir_load

    def save(self, epoch, model, best):
        if best == True:
            torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (self.dir_save, epoch))

        return None

    def load(self, filename):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            model = torch.load(filename)
        else:
            print("=> no checkpoint found at '{}'".format(filename))

        return model


class Counter:
    def __init__(self):
        self.mask_size = 0

    def update(self, size):
        self.mask_size += size

    def get_total(self):
        return self.mask_size


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
    else:
        print('\n\nActivation function {} is not supported/understood\n\n'.format(act))
        act_ = None
    return act_