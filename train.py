# train.py

import torch
import torch.optim as optim
import numpy as np
import math


class Trainer():
    def __init__(self, args, model, loss_fn):

        self.args = args
        self.model = model
        self.loss_fn = loss_fn
        self.dir_save = args.save
        self.cuda = args.cuda
        self.nepochs = args.nepochs
        self.nchannels = args.nchannels
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        self.momentum = args.momentum
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.weight_decay = args.weight_decay
        self.optim_method = args.optim_method
        self.dataset_train_name = args.dataset_train

        parameters = filter(lambda p: p.requires_grad, model.parameters())
        if self.optim_method == 'Adam':
            self.optimizer = optim.Adam(parameters, lr=self.lr, betas=(self.adam_beta1, self.adam_beta2), weight_decay=self.weight_decay)
        elif self.optim_method == 'RMSprop':
            self.optimizer = optim.RMSprop(parameters, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.optim_method == 'SGD':
            self.optimizer = optim.SGD(parameters, lr=self.lr,  momentum=self.momentum, weight_decay=self.weight_decay, nesterov=True)
        else:
            raise(Exception("Unknown Optimization Method"))


    def learning_rate(self, epoch):
        if self.dataset_train_name == 'CIFAR10':
            return self.lr * ((0.2 ** int(epoch >= 60)) * (0.2 ** int(epoch >= 90)) * (0.2 ** int(epoch >= 120)) * (0.2 ** int(epoch >= 160)))
        elif self.dataset_train_name == 'CIFAR100':
            return self.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 120))* (0.1 ** int(epoch >= 160)))
        elif self.dataset_train_name == 'MNIST':
            return self.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 120))* (0.1 ** int(epoch >= 160)))
        elif self.dataset_train_name == 'FRGC':
            return self.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 120))* (0.1 ** int(epoch >= 160)))
        elif self.dataset_train_name == 'ImageNet':
            decay = math.floor((epoch - 1) / 30)
            return self.lr * math.pow(0.1, decay)

    def get_optimizer(self, epoch, optimizer):
        lr = self.learning_rate(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def train(self, epoch, dataloader):
        self.optimizer = self.get_optimizer(epoch+1, self.optimizer)
        # switch to train mode
        self.model.train()

        losses = []
        accuracies = []
        for i, (input, label) in enumerate(dataloader):
            if self.args.cuda:
                label = label.cuda()
                input = input.cuda()

            output = self.model(input)
            loss = self.loss_fn(output, label)

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
                if self.args.cuda:
                    label = label.cuda()
                    input = input.cuda()

                output = self.model(input)
                loss = self.loss_fn(output,label)

                pred = output.data.max(1)[1]
                acc = pred.eq(label.data).cpu().sum()*100.0 / self.batch_size
                losses.append(loss.item())
                accuracies.append(acc)

        return np.mean(losses), np.mean(accuracies)
