# main.py

import torch
import random
from model import Model
from config import parser
from dataloader import Dataloader
from checkpoints import Checkpoints
from train import Trainer
import utils
from datetime import datetime

# parse the arguments
args = parser.parse_args()
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
utils.saveargs(args)

# initialize the checkpoint class
checkpoints = Checkpoints(args)

# Create Model
models = Model(args)
model, loss_fn = models.setup(checkpoints)


print('\n\n****** Model Configuration ******\n\n')
for arg in vars(args):
    print(arg, getattr(args, arg))

print('\n\nModel parameters:\n')
for name, param in model.named_parameters():
    #if param.requires_grad:
    print('{}  {}  {}  {:.2f}M'.format(name, list(param.size()), param.requires_grad, param.numel()/1000000.))

print('\n\nModel: {}, {:.2f}M parameters\n\n'.format(args.net_type, sum(p.numel() for p in model.parameters()) / 1000000.))

# Data Loading
dataloader = Dataloader(args)
loader_train, loader_test = dataloader.create()

# The trainer handles the training loop and evaluation on validation set
trainer = Trainer(args, model, loss_fn)

# start training !!!
acc_best = 0
print('\n\nTraining Model\n\n')
for epoch in range(args.nepochs):

    # train for a single epoch
    tr_loss, tr_acc = trainer.train(epoch, loader_train)
    te_loss, te_acc = trainer.test(epoch, loader_test)

    if te_acc > acc_best and epoch > 50:
        print('{}  Epoch {:d}/{:d}  Train: Loss {:.2f} Accuracy {:.2f} Test: Loss {:.2f} Accuracy {:.2f} (best result, saving to {})'.format(
                        str(datetime.now())[:-7], epoch, args.nepochs, tr_loss, tr_acc, te_loss, te_acc, args.save))
        model_best = True
        acc_best = te_acc
        checkpoints.save(epoch, model, model_best)
    else:
        print('{}  Epoch {:d}/{:d}  Train: Loss {:.2f} Accuracy {:.2f} Test: Loss {:.2f} Accuracy {:.2f}'.format(
                                str(datetime.now())[:-7], epoch, args.nepochs, tr_loss, tr_acc, te_loss, te_acc))