# main.py

import torch
import random
from model import Model
from dataloader import Dataloader
#from checkpoints import Checkpoints
#from train import Trainer
import utils
import os
from datetime import datetime
import argparse

result_path = "results/"
result_path = os.path.join(result_path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S/'))

parser = argparse.ArgumentParser(description='Your project title goes here')

# ======================== Data Setings ============================================
parser.add_argument('--dataset-test', type=str, default='CIFAR10', metavar='', help='name of training dataset')
parser.add_argument('--dataset-train', type=str, default='CIFAR10', metavar='', help='name of training dataset')
parser.add_argument('--split_test', type=float, default=None, metavar='', help='percentage of test dataset to split')
parser.add_argument('--split_train', type=float, default=None, metavar='', help='percentage of train dataset to split')
parser.add_argument('--dataroot', type=str, default='./data', metavar='', help='path to the data')
parser.add_argument('--save', type=str, default=result_path +'Save', metavar='', help='save the trained models here')
parser.add_argument('--logs', type=str, default=result_path +'Logs', metavar='', help='save the training log files here')
parser.add_argument('--resume', type=str, default=None, metavar='', help='full path of models to resume training')
parser.add_argument('--input-filename-test', type=str, default=None, metavar='', help='input test filename for filelist and folderlist')
parser.add_argument('--label-filename-test', type=str, default=None, metavar='', help='label test filename for filelist and folderlist')
parser.add_argument('--input-filename-train', type=str, default=None, metavar='', help='input train filename for filelist and folderlist')
parser.add_argument('--label-filename-train', type=str, default=None, metavar='', help='label train filename for filelist and folderlist')
parser.add_argument('--loader-input', type=str, default=None, metavar='', help='input loader')
parser.add_argument('--loader-label', type=str, default=None, metavar='', help='label loader')

# ======================== Network Model Setings ===================================
feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument('--perturb', dest='perturb', action='store_true')
feature_parser.add_argument('--no-perturb', dest='perturb', action='store_false')
parser.set_defaults(perturb=False)

parser.add_argument('--first_conv', type=int, default=0, metavar='', help='use conv layer with this kernel size in FirstLayer')
parser.add_argument('--nblocks', type=int, default=10, metavar='', help='number of blocks in each layer')
parser.add_argument('--nlayers', type=int, default=6, metavar='', help='number of layers')
parser.add_argument('--nchannels', type=int, default=3, metavar='', help='number of input channels')
parser.add_argument('--nfilters', type=int, default=64, metavar='', help='number of filters in each layer')
parser.add_argument('--nmasks', type=int, default=32, metavar='', help='number of noise masks per input channel (fan out)')
parser.add_argument('--level', type=float, default=0.5, metavar='', help='noise level for uniform noise')
parser.add_argument('--nunits', type=int, default=None, metavar='', help='number of units in hidden layers')
parser.add_argument('--dropout', type=float, default=None, metavar='', help='dropout parameter')
parser.add_argument('--net-type', type=str, default='resnet18', metavar='', help='type of network')
parser.add_argument('--length-scale', type=float, default=None, metavar='', help='length scale')
parser.add_argument('--tau', type=float, default=None, metavar='', help='Tau')

# ======================== Training Settings =======================================
parser.add_argument('--cuda', type=bool, default=True, metavar='', help='run on gpu')
parser.add_argument('--batch-size', type=int, default=64, metavar='', help='batch size for training')
parser.add_argument('--nepochs', type=int, default=500, metavar='', help='number of epochs to train')
parser.add_argument('--niters', type=int, default=None, metavar='', help='number of iterations at test time')
parser.add_argument('--epoch-number', type=int, default=None, metavar='', help='epoch number')
parser.add_argument('--nthreads', type=int, default=2, metavar='', help='number of threads for data loading')
parser.add_argument('--manual-seed', type=int, default=1, metavar='', help='manual seed for randomness')
parser.add_argument('--print_freq', type=int, default=100, metavar='', help='print results every print_freq batches')

# ======================== Hyperparameter Setings ==================================
parser.add_argument('--optim-method', type=str, default='SGD', metavar='', help='the optimization routine ')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='', help='learning rate')
parser.add_argument('--learning-rate-decay', type=float, default=None, metavar='', help='learning rate decay')
parser.add_argument('--momentum', type=float, default=0.9, metavar='', help='momentum')
parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='', help='weight decay')
parser.add_argument('--adam-beta1', type=float, default=0.9, metavar='', help='Beta 1 parameter for Adam')
parser.add_argument('--adam-beta2', type=float, default=0.999, metavar='', help='Beta 2 parameter for Adam')

args = parser.parse_args()
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
utils.saveargs(args)

checkpoints = utils.Checkpoints(args)
print(checkpoints)

setup = Model(args, checkpoints)
model = setup.model
train = setup.train
test = setup.test

print('\n\n****** Model Configuration ******\n\n')
for arg in vars(args):
    print(arg, getattr(args, arg))

print('\n\n****** Model Graph ******\n\n')
for arg in vars(model):
    print(arg, getattr(model, arg))

print('\n\nModel parameters:\n')
for name, param in model.named_parameters():
    #if param.requires_grad:
    print('{}  {}  {}  {:.2f}M'.format(name, list(param.size()), param.requires_grad, param.numel()/1000000.))

print('\n\nModel: {}, {:.2f}M parameters\n\n'.format(args.net_type, sum(p.numel() for p in model.parameters()) / 1000000.))

dataloader = Dataloader(args, setup.input_size)
loader_train, loader_test = dataloader.create()

acc_best = 0
print('\n\nTraining Model\n\n')
for epoch in range(args.nepochs):

    # train for a single epoch
    tr_loss, tr_acc = train(epoch, loader_train)
    te_loss, te_acc = test(loader_test)

    if te_acc > acc_best and epoch > 50:
        print('{}  Epoch {:d}/{:d}  Train: Loss {:.2f} Accuracy {:.2f} Test: Loss {:.2f} Accuracy {:.2f} (best result, saving to {})'.format(
                        str(datetime.now())[:-7], epoch, args.nepochs, tr_loss, tr_acc, te_loss, te_acc, args.save))
        model_best = True
        acc_best = te_acc
        checkpoints.save(epoch, model, model_best)
    else:
        if epoch == 0:
            print('\n')
        print('{}  Epoch {:d}/{:d}  Train: Loss {:.2f} Accuracy {:.2f} Test: Loss {:.2f} Accuracy {:.2f}'.format(
                                str(datetime.now())[:-7], epoch, args.nepochs, tr_loss, tr_acc, te_loss, te_acc))