# model.py

import math
import numpy as np
import torch
from torch import nn
import models
import torch.optim as optim


class Model:
    def __init__(self, args, checkpoints):
        self.cuda = args.cuda
        self.lr = args.learning_rate
        self.dataset_train_name = args.dataset_train
        self.perturb = args.perturb
        self.nfilters = args.nfilters
        self.nchannels = args.nchannels
        self.nblocks = args.nblocks
        self.nlayers = args.nlayers
        self.batch_size = args.batch_size
        self.level = args.level
        self.net_type = args.net_type
        self.nmasks = args.nmasks
        self.first_conv = args.first_conv

        if self.dataset_train_name.startswith("CIFAR"):
            self.input_size = 32
            self.nclasses = 10
            if self.first_conv < 7:
                self.avgpool = 4
            elif self.first_conv == 7:
                self.avgpool = 1

        elif self.dataset_train_name.startswith("ImageNet"):
            self.nclasses = 1000
            self.input_size = 224
            if self.first_conv < 7:
                self.avgpool = 14  #TODO
            elif self.first_conv == 7:
                self.avgpool = 7

        elif self.dataset_train_name.startswith("MNIST"):
            self.nclasses = 10
            self.input_size = 28
            if self.first_conv < 7:
                self.avgpool = 14  #TODO
            elif self.first_conv == 7:
                self.avgpool = 7

        self.model = getattr(models, self.net_type)(
            nfilters=self.nfilters,
            avgpool=self.avgpool,
            nclasses=self.nclasses,
            nmasks=self.nmasks,
            level=self.level,
            first_conv=self.first_conv,
            perturb=self.perturb
        )

        self.loss_fn = nn.CrossEntropyLoss()

        if self.cuda:
            self.model = self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()

        self.initialize_model(checkpoints)

        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        if args.optim_method == 'Adam':
            self.optimizer = optim.Adam(parameters, lr=self.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)
        elif args.optim_method == 'RMSprop':
            self.optimizer = optim.RMSprop(parameters, lr=self.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optim_method == 'SGD':
            self.optimizer = optim.SGD(parameters, lr=self.lr,  momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
            """
            self.optimizer = optim.SGD([{'params': [param for name, param in self.model.named_parameters() if 'noise' not in name]},
                                        {'params': [param for name, param in self.model.named_parameters() if 'noise' in name], 'lr': self.lr * 10},
                                        ], lr=self.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True) #"""
        else:
            raise(Exception("Unknown Optimization Method"))


    def initialize_model(self, checkpoints):
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if checkpoints.latest('resume') is None:
            self.model.apply(weights_init)
        else:
            print('\n\nLoading model from saved checkpoint at {}\n\n'.format(checkpoints))
            self.model.load_state_dict(checkpoints.load(checkpoints.latest('resume')))

    def learning_rate(self, epoch):
        if self.dataset_train_name == 'CIFAR10':
            new_lr = self.lr * ((0.2 ** int(epoch >= 60)) * (0.2 ** int(epoch >= 90)) * (0.2 ** int(epoch >= 120)) * (0.2 ** int(epoch >= 160)))
        elif self.dataset_train_name == 'CIFAR100':
            new_lr = self.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 120))* (0.1 ** int(epoch >= 160)))
        elif self.dataset_train_name == 'MNIST':
            new_lr = self.lr * ((0.2 ** int(epoch >= 30)) * (0.2 ** int(epoch >= 60))* (0.2 ** int(epoch >= 90)))
        elif self.dataset_train_name == 'FRGC':
            new_lr = self.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 120))* (0.1 ** int(epoch >= 160)))
        elif self.dataset_train_name == 'ImageNet':
            decay = math.floor((epoch - 1) / 30)
            new_lr = self.lr * math.pow(0.1, decay)
            #print('\nReducing learning rate to {}\n'.format(new_lr))
        return new_lr


    def train(self, epoch, dataloader):
        self.model.train()

        lr = self.learning_rate(epoch+1)

        for param_group in self.optimizer.param_groups:
            #print(param_group)
            param_group['lr'] = lr

        losses = []
        accuracies = []
        for i, (input, label) in enumerate(dataloader):
            if self.cuda:
                label = label.cuda()
                input = input.cuda()

            output = self.model(input)
            loss = self.loss_fn(output, label)
            #print('\nBatch:', i)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pred = output.data.max(1)[1]

            acc = pred.eq(label.data).cpu().sum()*100.0 / self.batch_size

            losses.append(loss.item())
            accuracies.append(acc)

        return np.mean(losses), np.mean(accuracies)

    def test(self, dataloader):
        self.model.eval()
        losses = []
        accuracies = []
        with torch.no_grad():
            for i, (input, label) in enumerate(dataloader):
                if self.cuda:
                    label = label.cuda()
                    input = input.cuda()

                output = self.model(input)
                loss = self.loss_fn(output, label)

                pred = output.data.max(1)[1]
                acc = pred.eq(label.data).cpu().sum()*100.0 / self.batch_size
                losses.append(loss.item())
                accuracies.append(acc)

        return np.mean(losses), np.mean(accuracies)
