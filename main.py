import argparse
import os
import shutil
import time
import sys
import collections
from collections import defaultdict
from functools import partial

from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
from utils import *
from representation import _get_pwcca_avg_pool, _get_pwcca_baseline, _get_RV_avg_pool

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

activations = {}

'''
10x10 Class Activation Similarity Matrix for CIFAR10 dataset.
Initially no relation assumed between classes. Each is initialized to 0.1.
'''
AS_fc = F.softmax(torch.ones([10, 10], dtype=torch.float32), dim=1).cuda()
AS_highest = F.softmax(torch.ones([10, 10], dtype=torch.float32), dim=1).cuda()
AS_high = F.softmax(torch.ones([10, 10], dtype=torch.float32), dim=1).cuda()
AS_mid_high = F.softmax(torch.ones([10, 10], dtype=torch.float32), dim=1).cuda()
AS_mid = F.softmax(torch.ones([10, 10], dtype=torch.float32), dim=1).cuda()
AS_mid_low = F.softmax(torch.ones([10, 10], dtype=torch.float32), dim=1).cuda()
AS_val = F.softmax(torch.ones([10, 10], dtype=torch.float32), dim=1).cuda()

AS_updated = []
first_update = False
update_patience = 4
counter = 0
should_update = False
is_best = False
gen_error = []

def get_activation(name, model, input, output):
    activations[name] = output.detach().cpu()

layer_activations = defaultdict()
#layer_activations_prev = defaultdict(list)

'''
mean_cca_list = defaultdict(list)
baseline_cca_list = defaultdict(list)
'''

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: vgg19)')
parser.add_argument('--resnet-width', '--rw', default=64,
                    help='Resnet18 width. (default: 64)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('-l', '--label-noise', default=0, type=int,
                     help='\%of label noise in training set (default: 0\%)')
parser.add_argument('-o', '--optimizer', dest="optim", default="SGD",
                    type=str, help="Optimizer. Options: SGD, Adam")
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument("--patiance", "--pt", default=0, type=int,
            help="Patiance value for early stopping. Default: 0 (No early stopping)")
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--activation-dir', dest='act_dir',
                    help='The directory used to save the layer activations',
                    default='activations', type=str)
parser.add_argument('--similarity-metric', dest='sim_met',
                    help='Similarity metric for layer-wise comparison',
                    default='RV', type=str)
parser.add_argument('--collect-cosine', dest='collect_cosine', action='store_true',
                    help='Collect cosine distances of weights to their initial versions')
parser.add_argument('--collect-acts', dest='collect_activations', action='store_true',
                    help='Collect activations to their initial versions')
parser.add_argument('-r', '--refer-initial', dest='refer_init', action='store_true',
                    help='Collect activations that compare to their checkpoint versions OR \
                    cosine distances that refer to initial random weights')
parser.add_argument('--assert-lr', dest="assert_lr", action='store_true',
                    help='Assert larger initial learning rate when resuming, as opposed to \
                    checkpoint adjustment.')
parser.add_argument('--constant-lr', dest="constant_lr", action='store_true',
                    help='Constant learning rate.')

best_prec1 = 0

def main():
    global args, best_prec1, gen_error, is_best, layer_activations, activations
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = models.__dict__[args.arch]()

    #model.print_arch()
    #sys.exit()

    if args.cpu:
        model.cpu()
    else:
        model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

        if args.collect_cosine and not args.refer_init:
            model.update_reference_weights()

    cudnn.benchmark = True
    
    if args.patiance != 0:
        early_stopping = EarlyStopping(patience=args.patiance, verbose=True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_set = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)

    if args.label_noise > 0:
        set_size = len(train_set.targets)
        percentage = args.label_noise + args.label_noise/9.0
        random_indexes = np.random.choice(set_size, int(set_size*percentage/100.0), replace=False)
        random_labels = np.random.choice(10, len(random_indexes), replace=True)
        i = 0
        for idx in random_indexes:
            train_set.targets[idx] = random_labels[i]
            i += 1

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    # define loss function (criterion) and optimizer
    #criterion = nn.KLDivLoss(reduction='batchmean')
    criterion = nn.CrossEntropyLoss()
    
    if args.cpu:
        criterion = criterion.cpu()
    else:
        criterion = criterion.cuda()

    if args.half:
        model.half()
        criterion.half()

    if args.optim.lower() == "sgd": 
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optim.lower() == "adam":
        if args.lr == 0.05:
            args.lr = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    else:
        raise ValueError("Invalid optimizer choice: \"{0}\". Options are: \"SGD\", \"Adam\"".format(args.optim))

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    if args.collect_cosine:
        param_keys = []
        distances_list = [[] for i in range(len(list(model.buffers())))]
    
    acc_train_list = []
    acc_val_list = []
    loss_train_list = []
    loss_val_list = []
    gen_error = [1.0]
        
    for epoch in range(args.start_epoch, args.epochs):
        if not args.constant_lr:
            adjust_learning_rate(optimizer, epoch, AS_updated)

        # train for one epoch
        if collect_activations:
            layer_activations.clear()
            activations.clear()

        prec1_train, loss_train = train(train_loader, model, criterion, optimizer, epoch)        

        # evaluate on validation set
        prec1, loss_val = validate(val_loader, model, criterion, epoch)
        
        curr_gen_error_ratio = np.abs(loss_train/loss_val)
        gen_error.append(curr_gen_error_ratio)
        plot_AS(epoch, curr_gen_error_ratio, prec1_train, prec1)

        # append loss and prec1 values
        acc_train_list.append(prec1_train)
        acc_val_list.append(prec1)
        loss_train_list.append(loss_train)
        loss_val_list.append(loss_val)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if args.collect_cosine:
            if is_best:
                print("Cosine distances:")
            param_keys, distances = model.get_cosine_distance_per_layer()
            for idx, p_key in enumerate(param_keys):
                if is_best:
                    print("    {0}:".format(p_key), distances[idx])
                distances_list[idx].append(distances[idx])

        if is_best:
           save_checkpoint({
               'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))
        
        if args.patiance != 0:
            early_stopping(loss_val, model)
            if early_stopping.early_stop == True:
                break

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plot_train_val_accs(acc_train_list, acc_val_list, timestamp)
    plot_loss(loss_train_list, loss_val_list, timestamp)

    if args.collect_cosine:
        plot_cosine_distances(param_keys, distances_list, timestamp)

    #if args.collect_activations:
    #    plot_similarities(timestamp)


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    global AS_val

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    
    end = time.time()
    
    if first_update:
        beta = 1.0
    else:
        beta = 1.0 # 0.5 + 1.0/(epoch//3) #get_beta(avg_gen_error_ratio)
    print("New BETA value: {}".format(beta))

    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if args.cpu == False:
            input = input.cuda(async=True)
            target = target.cuda(async=True)
        if args.half:
            input = input.half()

        # compute output
        output = model(input)
        
        ## Required conversions for KL loss.
        '''
        output_logp = F.log_softmax(output, dim=1)
        one_hot_t = to_one_hot(target, 10).cuda().float()
        activation_sim_dist = torch.index_select(AS_val, 0, target).cuda().float()
        target_dist = (beta*one_hot_t + (1-beta)*activation_sim_dist)

        if i==0:
            print("Target", one_hot_t[2])
            print("AS dist.", activation_sim_dist[2])
            print("Interpolation", target_dist[2])
            print("Output", F.softmax(output, dim=1)[2])
        '''
        
        #loss = criterion(output_logp.cuda(async=True), target_dist.cuda(async=True))

        loss = criterion(output, target)

        '''
        if args.collect_activations:
            for k,v in activations.items():
                # The number of datapoints for each layer's activation output is determined as approximately 2500.
                # CCA methods are sensitive to dimensionality hence require lots of data points.
                # RV is less sensitive, so in case RV is chosen, less might suffice.
                if args.batch_size*(i+1) > 600:
                    break

                if epoch == 0:
                    layer_activations_prev[k].append(v.detach().cpu().numpy())
                else:
                    layer_activations[k].append(v.detach().cpu().numpy())
            
            activations.clear()
        '''

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

    
    return top1.avg, losses.avg

def validate(val_loader, model, criterion, epoch):
    """
    Run evaluation
    """
    global activations
    global AS_updated
    global update_patience
    global counter
    global gen_error    
    global first_update
    global is_best
    global layer_activations
    global AS_val
    #global layer_activations_prev


    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    avg_gen_error_ratio = (np.array(gen_error, dtype=np.float32)[-update_patience:].mean())
    curr_gen_error_ratio = gen_error[-1]

    print(" * CURRENT GEN-ERR-RATIO: {}".format(curr_gen_error_ratio))
    print(" * AVG GEN-ERR-RATIO: {}".format(avg_gen_error_ratio))

    counter += 1
    first_update = (len(AS_updated) == 0) or epoch < 10
    should_update_val = (avg_gen_error_ratio>0.90 and counter == update_patience) or epoch == 10
    should_update = (curr_gen_error_ratio>0.01) and epoch >= 0 #(curr_gen_error_ratio>0.90) and epoch >= 4
    #should_downgrade = (avg_gen_error_ratio<0.80) and (counter) == update_patience
    
    if should_update_val:
        counter = 0
        AS_updated.append(epoch)
        #AS_val = AS
        print("Generalization gap DECREASED. Activation Similarity matrix will be updated.")
    #elif should_downgrade:
    #    counter = 0
    #    AS_updated.append(epoch)
    #    print("Generalization gap INCREASED. Activation Similarity matrix will be downgraded.")

    if args.collect_activations:
        hooks = []
        if args.evaluate:
            for name, m in model.named_modules():
                if type(m)==nn.Conv2d or type(m)==nn.Linear:
                    # partial to assign the layer name to each hook
                    hooks.append(m.register_forward_hook(partial(get_activation, name)))

        else:
            if should_update:
                for name, m in model.named_modules():
                    #if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv2d):
                    #if type(m)==nn.Conv2d or type(m)==nn.Linear:

                    # Register the last 3 convolutional layer activations before linear layers.
                    # These activations seem to be sensitive to the relation between 
                    #   generlization error and accuracy: When increase in the accuracy slows down
                    #   generalization gap starts to widen. Also, these layers are most likely to
                    #   yield high level shared features between classes. 

                    ## THIS WORKS ONLY WITH VGG19 with Batch Norm.
                    ## Layer indexes needs to be inspected for different architectures.
                    splt = name.split(".")
                    if len(splt) == 2:
                        sub_name, indx = splt
                        if type(m)==nn.Conv2d and sub_name == "features": # and int(indx) in [3,7,10,14,17,20,23,40, 43, 46, 49]:
                            # partial to assign the layer name to each hook
                            hooks.append(m.register_forward_hook(partial(get_activation, name)))
                        elif sub_name == "classifier" and int(indx) in [1,4]:
                            hooks.append(m.register_forward_hook(partial(get_activation, name)))

    if first_update:
        beta = 1.0
    else:
        beta = 1.0 # 0.5 + 1.0/(epoch//3) #get_beta(avg_gen_error_ratio)
    
    for i, (input, target) in enumerate(val_loader):

        if args.cpu == False:
            input = input.cuda(async=True)
            target = target.cuda(async=True)

        if args.half:
            input = input.half()

        # compute output
        with torch.no_grad():
            output = model(input)
            '''
            ## Required conversions for KL loss.
            output_logp = F.log_softmax(output, dim=1)
            
            one_hot_t = to_one_hot(target, 10).cuda().float()
            activation_sim_dist = torch.index_select(AS_val, 0, target).cuda().float()            
            target_dist = (beta*one_hot_t + (1-beta)*activation_sim_dist)
            
            loss = criterion(output_logp.cuda(async=True), target_dist.cuda(async=True))
            '''
            loss = criterion(output, target)

        if args.collect_activations:
            if args.batch_size*(i+1) > 5000: 
                if len(activations):
                    activations.clear()
                    for h in hooks:
                        h.remove()
            else:
                if args.evaluate:
                    for k,v in activations.items():
                        dest = os.path.join(args.act_dir, "{0}_{1}.npy".format(k, i))
                        np.save(dest, v.numpy())
                
                elif should_update: #not should_downgrade and epoch >= 10:
                    for k,v in activations.items():
                        v_np = v.detach().cpu().numpy()
                        for c in range(10):
                            if c not in layer_activations.keys():
                                 layer_activations[c] = defaultdict(list)
                            class_indexes = (target.cpu().numpy() == c)
                            layer_activations[c][k].append(v_np[class_indexes])
            

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    # Update the Activations similarity matrix if the generalization error is above 0.9.
    #if should_downgrade:
    #    AS = AS_val
    if should_update and args.collect_activations:
        update_AS()
    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))


    return top1.avg, losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

def adjust_learning_rate(optimizer, epoch, AS_updated):
    """Sets the learning rate to initial-rate * 1/sqrt(epoch)"""
    if epoch == 0:
        return
    if args.resume and args.assert_lr:
        #lr = args.lr * (0.5 ** ((epoch - args.start_epoch) // 30))  
        lr = args.lr / np.sqrt(epoch - args.start_epoch)
    else:  
        #lr = args.lr * (0.5 ** (epoch // 30))
        lr = args.lr / np.sqrt(epoch-len(AS_updated)+1)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def update_AS():
    global layer_activations
    global AS_fc, AS_highest, AS_high, AS_mid_high, AS_mid, AS_mid_low 

    for c1 in range(10):
        for c2 in range(c1, 10):
            c1_act_list = layer_activations[c1]
            c2_act_list = layer_activations[c2]
            similarity_fc = 0.0
            similarity_highest = 0.0
            similarity_high = 0.0
            similarity_mid_high = 0.0
            similarity_mid = 0.0
            similarity_mid_low = 0.0

            for k, c1_act_o in c1_act_list.items():
            
                c1_act = c1_act_o[:len(c1_act_o)//2]
                c2_act = c2_act_list[k][len(c2_act_list[k])//2:]

                splt = k.split(".")
                sub_name, indx = splt
                if sub_name == "features":
                    if int(indx) in [20,23]:#[3,7,10]:
                        similarity_mid_low += _get_RV_avg_pool(np.vstack(c1_act), np.vstack(c2_act), 512)
                    elif int(indx) in [27,30]:#[14,17,20,23]:
                        similarity_mid += _get_RV_avg_pool(np.vstack(c1_act), np.vstack(c2_act), 512)
                    elif int(indx) in [33,36]:#[27,30,33,36]:
                        similarity_mid_high += _get_RV_avg_pool(np.vstack(c1_act), np.vstack(c2_act), 512)
                    elif int(indx) in [40,43]:
                        similarity_high += _get_RV_avg_pool(np.vstack(c1_act), np.vstack(c2_act), 512)
                    elif int(indx) in [46,49]:
                        similarity_highest += _get_RV_avg_pool(np.vstack(c1_act), np.vstack(c2_act), 512)
                else:
                    similarity_fc +=_get_RV_avg_pool(np.vstack(c1_act), np.vstack(c2_act), 512)
            
            AS_fc[c1,c2] = similarity_fc/2
            AS_highest[c1,c2] = similarity_highest/2
            AS_high[c1,c2] = similarity_high/2
            AS_mid_high[c1,c2] = similarity_mid_high/2
            AS_mid[c1,c2] = similarity_mid/2
            AS_mid_low[c1,c2] = similarity_mid_low/2
    
    for c1 in range(10):
        for c2 in range(c1):
                AS_fc[c1,c2] = AS_fc[c2,c1]
                AS_highest[c1,c2] = AS_highest[c2,c1]
                AS_high[c1,c2] = AS_high[c2,c1]
                AS_mid_high[c1,c2] = AS_mid_high[c2,c1]
                AS_mid[c1,c2] = AS_mid[c2,c1]
                AS_mid_low[c1,c2] = AS_mid_low[c2,c1]
    
    layer_activations.clear()
    #print(AS)
    #for i in range(10):
    #    mean = torch.mean(AS[i,:])
    #    std = torch.std(AS[i,:])
    #    AS[i,:] = (AS[i,:]-mean)/std

    #AS_high = F.softmax(AS, dim=1)
    #if first_update:
    #    AS_val = AS_high


def plot_AS(epoch, curr_gen_error_ratio, acc_train, acc_val):
    plt.clf()
    ax1 = sns.heatmap(AS_mid_low.cpu().numpy(), vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
    plt.savefig("figures/class_sim/{4}_class_sim_MID_LOW_EPOCH-{0}_gen-error-{1:.2f}__{2:.1f}-{3:.1f}.png".format(epoch, curr_gen_error_ratio, acc_train, acc_val, args.arch))
    plt.clf()
    ax2 = sns.heatmap(AS_mid.cpu().numpy(), vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
    plt.savefig("figures/class_sim/{4}_class_sim_MID_EPOCH-{0}_gen-error-{1:.2f}__{2:.1f}-{3:.1f}.png".format(epoch, curr_gen_error_ratio, acc_train, acc_val, args.arch))
    plt.clf()
    ax3 = sns.heatmap(AS_mid_high.cpu().numpy(), vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
    plt.savefig("figures/class_sim/{4}_class_sim_MID_HIGH_EPOCH-{0}_gen-error-{1:.2f}__{2:.1f}-{3:.1f}.png".format(epoch, curr_gen_error_ratio, acc_train, acc_val, args.arch))
    plt.clf()
    ax4 = sns.heatmap(AS_high.cpu().numpy(), vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
    plt.savefig("figures/class_sim/{4}_class_sim_HIGH_EPOCH-{0}_gen-error-{1:.2f}__{2:.1f}-{3:.1f}.png".format(epoch, curr_gen_error_ratio, acc_train, acc_val, args.arch))
    plt.clf()
    ax5 = sns.heatmap(AS_highest.cpu().numpy(), vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
    plt.savefig("figures/class_sim/{4}_class_sim_HIGHEST_EPOCH-{0}_gen-error-{1:.2f}__{2:.1f}-{3:.1f}.png".format(epoch, curr_gen_error_ratio, acc_train, acc_val, args.arch))
    plt.clf()
    ax6 = sns.heatmap(AS_fc.cpu().numpy(), vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
    plt.savefig("figures/class_sim/{4}_class_sim_FC_EPOCH-{0}_gen-error-{1:.2f}__{2:.1f}-{3:.1f}.png".format(epoch, curr_gen_error_ratio, acc_train, acc_val, args.arch))


def plot_cosine_distances(param_keys, distance_list, timestamp):
    """ Plots cosine distance values of layer weights (w.r.t intialization values)"""

    plt.figure()
    plt.grid()
    save_path = "figures/cosine_distance/cosine_distance_per_epoch_arch-{0}_{1}.png".format(args.arch, timestamp)
    data_path = "data/cosine_distance/cosine_distance_per_epoch_arch-{0}_{1}.pickle".format(args.arch, timestamp)

    '''
    default_cycler = (cycler(color=list('rgbycmk')) *
                  cycler(linestyle=['-', '--', ':', '-.']))

    plt.rc('axes', prop_cycle=default_cycler)

    '''
    
    c = np.arange(1, len(param_keys) + 1)
    epoch_axis = np.arange(len(distance_list[0]))

    title = "Cosine distance"

    if args.resume:
        last_epoch = args.start_epoch
        if args.assert_lr:
            title += ", High learning rate"
        epoch_axis = epoch_axis + last_epoch
        if args.refer_init:
            title += ", Re-randomized"

    if args.constant_lr:
        title += ", Constant Learning Rate" 

    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.inferno)
    cmap.set_array([])

    plt.xlabel("Epoch")
    plt.ylabel("Cosine Distance")
    for idx, p_key in enumerate(param_keys):
        plt.plot(epoch_axis, distance_list[idx], c=cmap.to_rgba(idx+1))
    #plt.legend(loc='lower right')

    cbar = plt.colorbar(cmap, ticks=[1,len(param_keys)])
    cbar.set_label("Layers")
    plt.title(title)
    plt.savefig(save_path)
    plt.clf()

    final_dict = vars(args)
    final_dict["distance_list"] = distance_list
    save_dictionary_to_file(data_path, final_dict)

def plot_similarities(timestamp):
    """ Plots activation similarities that we kept track of during training"""
    
    global mean_cca_list
    global baseline_cca_list

    save_path = "figures/inLayer-similarity/{2}_{1}_inLayer-similarity_wrt-{0}_{3}.png".format(
        ("initial" if args.refer_init else "prev"), k, args.arch)
    
    for k,v in mean_cca_list.items():
        v_baseline = baseline_cca_list[k]
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Similarity")
        plt.grid()
        plt.plot(v, label=args.sim_met)
        plt.plot(v_baseline, label='Baseline')
        plt.legend(loc='upper right')
        plt.savefig(save_path)
        plt.clf()

    final_dict = vars(args)
    final_dict["similarity_list"] = mean_cca_list
    final_dict["baseline_list"] = baseline_cca_list
    data_path = "data/inlayer_similarity/{1}_inLayer-similarities_{0}.pickle".format(timestamp, args.arch)
    save_dictionary_to_file(data_path, final_dict)

def plot_train_val_accs(train_accs, val_accs, timestamp):
    """  Plots training and validation accuarcy values in a single figure"""

    save_path = "figures/acc/accs_per_epoch_a-{0}_{1}.png".format(args.arch, timestamp)
    
    plt.figure()
    plt.grid()
    title = "Acurracy"
    epoch_axis = np.arange(len(train_accs))
    if args.resume:
        epoch_axis += args.start_epoch
        if args.assert_lr:
            title += ", High learning rate"
        if args.refer_init:
            title += ", Re-randomized"

    if args.constant_lr:
        title += ", Constant learning rate" 

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.plot(epoch_axis, train_accs, label='train_acc')
    plt.plot(epoch_axis, val_accs, label='val_acc')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.savefig(save_path)
    plt.clf()

def plot_loss(train_loss, val_loss, timestamp):
    """  Plots training and validation loss history in a single figure"""

    save_path = "figures/loss/loss_per_epoch_a-{0}_{1}.png".format(args.arch, timestamp)
    data_path = "data/loss_history/loss_per_epoch_a-{0}_{1}.npy".format(args.arch, timestamp)
    
    plt.figure()
    title = "Loss"
    epoch_axis = np.arange(len(train_loss))
    if args.resume:
        epoch_axis += args.start_epoch
        if args.assert_lr:
            title += ", High learning rate"
        if args.refer_init:
            title += ", Re-randomized"

    if args.constant_lr:
        title += ", Constant learning rate" 

    plt.plot(epoch_axis, train_loss, label='train_loss')
    plt.plot(epoch_axis, val_loss, label='val_loss')
    plt.legend(loc='upper right')
    plt.title(title)
    plt.savefig(save_path)
    plt.clf()

    final_dict = vars(args)
    final_dict["train_loss"] = train_loss
    final_dict["val_loss"] = val_loss
    save_dictionary_to_file(data_path, final_dict)


def calculate_sim_for_epoch():
    """ calculates similarity of the activations in each layer compared to previous ones """
    
    global layer_activations, layer_activations_prev

    for k,v in layer_activations.items():
        act_current = np.vstack(v)
        act_prev = np.vstack(layer_activations_prev[k])
        
        # depending on the metric calculate baseline and original similarity for each layer
        if method == "RV":
            mean_cca = _get_pwcca_avg_pool(act_current, act_prev, 500)
            baseline_cca = _get_pwcca_baseline(act_current, act_prev, 500)
        elif method == "pwcca":
            mean_cca = _get_pwcca_avg_pool(act_current, act_prev, 2500)
            baseline_cca = _get_pwcca_baseline(act_current, act_prev, 2500)
        
        mean_cca_list[k].append(mean_cca)
        baseline_cca_list[k].append(baseline_cca)

    # if refer_init, we keep track similarity of activations w.r.t first epoch activations.
    if args.refer_init:
        layer_activations.clear()
    # else we take previous epoch as refernce
    else:
        layer_activations_prev.clear()
        layer_activations_prev = layer_activations
        layer_activations = defaultdict(list)

if __name__ == '__main__':
    main()
