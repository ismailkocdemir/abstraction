import argparse
import os
import shutil
import time
import sys
import collections
from collections import defaultdict, Counter
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
from autograd_hacks import *
from utils import *
from representation import _get_pwcca_avg_pool, _get_pwcca_baseline, _get_RV_avg_pool, _get_RV_baseline, _get_linear_cka_baseline, _get_linear_cka_avg_pool

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

stiffness_avg_dict = defaultdict()
epoch_interval = 5
AS_avg_dict = defaultdict()
stiffness_data_size = 1024
activation_data_size = 1024

is_best = False

layer_activations = defaultdict()
layer_gradients = defaultdict()

best_prec1 = 0
prev_prec = 0

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: vgg19)')
parser.add_argument('--resnet-width', '--rw', default=64,
                    help='Resnet18 width. (default: 64)')
parser.add_argument('--dataset', '--d', type=str, default='cifar10',
                        help='dataset to train on.', choices=['cifar10', 'imagenet'])


parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')


## OPTIMIZATION
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
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
parser.add_argument('--kl', dest='kl', action='store_true',
                    help='use KL divergence as the objective')


parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--assert-lr', dest="assert_lr", action='store_true',
                    help='Assert larger initial learning rate when resuming, as opposed to \
                    checkpoint adjustment.')
parser.add_argument('--constant-lr', dest="constant_lr", action='store_true',
                    help='Constant learning rate.')

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
parser.add_argument('--grad-dir', dest='grad_dir',
                    help='The directory used to save the layer gradients',
                    default='gradients', type=str)


parser.add_argument('--collect-cosine', dest='collect_cosine', action='store_true',
                    help='Collect cosine distances of weights to their initial versions')
parser.add_argument('--collect-acts', dest='collect_activations', action='store_true',
                    help='Collect activations to their initial versions')
parser.add_argument('--stiffness', dest='stiffness', action='store_true',
                    help='Collect stiffness values for gradients')
parser.add_argument('--similarity-metric', dest='sim_met',
                    help='Similarity metric for layer-wise comparison',
                    default='RV', type=str)
parser.add_argument('-r', '--refer-initial', dest='refer_init', action='store_true',
                    help='Collect activations that compare to their checkpoint versions OR \
                    cosine distances that refer to initial random weights')

parser.add_argument('-l', '--label-noise', default=0, type=int,
                     help='\%of label noise in training set (default: 0\%)')



def main():
    global args, best_prec1, is_best, layer_activations, prev_prec
    args = parser.parse_args()
    dataset_choice = args.dataset.lower()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if dataset_choice == "cifar10":
        model = models.__dict__[args.arch](num_classes=10)
    else:
        model = models.__dict__[args.arch](num_classes=1000)

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

    if dataset_choice == 'cifar10':
        """
        if 'resnet18' == args.arch.lower().split("_")[0]:
            _transforms = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.Resize(224),
                            transforms.ToTensor(),
                            normalize,
                        ])
            _transforms_val = transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.ToTensor(),
                                    normalize,
                                ])
        else:
        """
        _transforms = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        normalize,
                    ])
        _transforms_val = _transforms_val = transforms.Compose([
                                transforms.ToTensor(),
                                normalize,
                            ])

        train_set = datasets.CIFAR10(root='./data', train=True, transform=_transforms, download=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=_transforms_val, download=True),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    else:
        train_set = datasets.ImageNet(
            root = './data', train=True,
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), download=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageNet(root="./data", split='val',
                trasform=trantransforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)


    if args.label_noise > 0:
        set_size = len(train_set.targets)
        percentage = args.label_noise + args.label_noise/9.0
        random_indexes = np.random.choice(set_size, int(set_size*percentage/100.0), replace=False)
        random_labels = np.random.choice(10, len(random_indexes), replace=True)
        i = 0
        for idx in random_indexes:
            train_set.targets[idx] = random_labels[i]
            i += 1

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # define loss function (criterion) and optimizer
    if args.kl:
        criterion = nn.KLDivLoss(reduction='batchmean')
    else:
        criterion = nn.CrossEntropyLoss(reduction='mean')
    
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
        validate(val_loader, model, criterion, 0)
        return

    if args.collect_cosine:
        param_keys = []
        distances_list = [[] for i in range(len(list(model.buffers())))]
    
    acc_train_list = []
    acc_val_list = []
    loss_train_list = []
    loss_val_list = []
    
    if args.start_epoch == 0:
       save_checkpoint({
           'epoch': 'baseline',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_baseline.tar'))
        
    for epoch in range(args.start_epoch, args.epochs):
        if not args.constant_lr and args.optim.lower() == "sgd":
            adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        if args.collect_activations:
            layer_activations.clear()

        prec1_train, loss_train = train(train_loader, model, criterion, optimizer, epoch)        

        # evaluate on validation set
        prec1, loss_val = validate(val_loader, model, criterion, epoch)

        if epoch >= epoch_interval  and epoch%10 == epoch_interval:
            if args.collect_activations:
                plot_AS(epoch)
            if args.stiffness:
                plot_stiffness(epoch)

        # append loss and prec1 values
        acc_train_list.append(prec1_train)
        acc_val_list.append(prec1)
        loss_train_list.append(loss_train)
        loss_val_list.append(loss_val)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        is_best_prev = prec1 > prev_prec

        prev_prec = prec1
        best_prec1 = max(prec1, best_prec1)

        if args.collect_cosine:
            if is_best:
                print("Cosine distances:")
            param_keys, distances = model.get_cosine_distance_per_layer()
            for idx, p_key in enumerate(param_keys):
                if is_best:
                    print("    {0}:".format(p_key), distances[idx])
                distances_list[idx].append(distances[idx])

        if is_best_prev:
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


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """

    global AS_val, epoch_interval, stiffness_data_size

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    should_update = epoch%epoch_interval == 0 # and epoch >= epoch_interval
    
    if args.stiffness and should_update:
        enable_hooks()
        add_hooks(model)

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if args.cpu == False:
            input = input.cuda(async=True)
            target = target.cuda(async=True)
        if args.half:
            input = input.half()

        # compute output
        optimizer.zero_grad()

        if args.stiffness and args.batch_size*(i+1) >= stiffness_data_size  and should_update:
            disable_hooks()
            remove_hooks(model)

        output = model(input)
        
        if args.kl:
            ## Required conversions for KL loss.
            output_logp = F.log_softmax(output, dim=1)
            one_hot_target = convert_to_one_hot(target, 10).cuda().float()
            loss = criterion(output_logp.cuda(async=True), one_hot_target.cuda(async=True))

        else:
            ## For CrossEntropy Loss
            loss = criterion(output, target)

        # compute gradient
        loss.backward()

        if args.stiffness and args.batch_size*(i+1) < stiffness_data_size and should_update:
            update_stiffness(model, target, i)

        #do SGD step
        optimizer.step()
        # measure accuracy and record loss
        output = output.float()
        loss = loss.float()
        
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
    global is_best, activation_data_size
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    should_update = epoch%epoch_interval == 0
    
    if args.collect_activations:
        if args.evaluate or should_update:
            enable_hooks()
            add_hooks(model, grad1=False)

    if not args.evaluate and (args.collect_activations or args.stiffness):
        final_activations_per_class = torch.zeros([10, 10], dtype=torch.float32).cuda()
        acc = 0

    for i, (input, target) in enumerate(val_loader):

        if args.cpu == False:
            input = input.cuda(async=True)
            target = target.cuda(async=True)

        if args.half:
            input = input.half()

        # compute output
        with torch.no_grad():
            output = model(input)
            
            if args.kl:
                ## Required conversions for KL loss.
                output_logp = F.log_softmax(output, dim=1)
                one_hot_target = convert_to_one_hot(target, 10).cuda().float()
                loss = criterion(output_logp.cuda(async=True), one_hot_target.cuda(async=True))
            
            else:
                ## For CrossEntropy Loss
                loss = criterion(output, target)

        if args.collect_activations:
            if args.batch_size*(i+1) > activation_data_size: 
                disable_hooks()
                remove_hooks(model)
            else:
                if args.evaluate:
                    save_activations(args.act_dir, i)
                
                elif should_update:
                    for k,v in get_activations().items():
                        if k not in layer_activations.keys():
                            layer_activations[k] = defaultdict(list)

                        v_np = v.detach().cpu().numpy()
                        for c in range(10):
                            class_indexes = (target.cpu().numpy() == c)
                            layer_activations[k][c].append(v_np[class_indexes])

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        #accumulate class activations
        if not args.evaluate and (args.collect_activations or args.stiffness): #and epoch==epoch_interval):
            acc +=1
            for c in range(10):
                class_indices = target == c
                x =  F.softmax(output[class_indices],dim=1)
                
                none_index = x!=x
                nan = torch.sum(none_index, dim=1)
                not_nan = nan == 0

                x_not_nan = x[not_nan]

                if x_not_nan.size()[0] != 0:
                    final_activations_per_class[c] += torch.mean(x_not_nan, dim=0)

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

    if not args.evaluate and (args.collect_activations or args.stiffness): # and epoch==epoch_interval):
        if args.collect_activations:
            update_AS(epoch)

        final_activations_per_class = final_activations_per_class/acc
        plt.clf()
        ax0 = sns.heatmap(final_activations_per_class.cpu().numpy(), vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
        
        save_dir = "figures/class_accuracy/{}/".format(args.arch)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig(os.path.join(save_dir, "{1}_final_classifier_EPOCH-{0}.png".format(epoch, args.arch)))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to initial-rate * 1/sqrt(epoch)"""
    if epoch == 0:
        return
    if args.resume and args.assert_lr:
        #lr = args.lr * (0.5 ** ((epoch - args.start_epoch) // 30))  
        lr = args.lr / np.sqrt(epoch - args.start_epoch)
    else:  
        #lr = args.lr * (0.5 ** (epoch // 30))
        lr = args.lr / np.sqrt(epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def update_stiffness(model, target, i):
    """ Compute the class similarity matices from the gradient stiffness values """
    global stiffness_avg_dict
    with torch.no_grad():
        compute_grad1(model, 'mean')

        for name, param in model.named_parameters():
            if 'bias' in name:
                continue

            dest = os.path.join(args.grad_dir, "{0}_{1}.npy".format(name, i))
            np.save(dest, (param.grad1 - param.grad).numpy())

            if name not in stiffness_avg_dict:
                stiffness_avg_dict[name] = AverageMeter(torch.zeros([10, 10], dtype=torch.float32).cuda(), keep_val=False)
        
            layer_class_sim = torch.zeros([10, 10], dtype=torch.float32).cuda()
            for c1 in range(10):
                for c2 in range(c1,10):
                    try:
                        assert hasattr(param, 'grad1'), "{} has no attribute named 'grad1'".format(name)
                        
                        # Subtract the mean gradient vector calculated from all classes, 
                        # hoping that this can help surpress the between-classes similarities and magnify
                        # the in-class similarities. In the layers that have no class specific information, 
                        # this should not change anything.                            
                        if c1 == c2:
                            class_indices = (target == c1)
                            class_grads = param.grad1[class_indices].cuda()
                            #class_grads -= param.grad
                            num_of_samples = len(class_grads)//2
                            grads1 = class_grads[:num_of_samples] #torch.flatten(class_grads[:num_of_samples], start_dim=1)
                            grads2 = class_grads[num_of_samples:2*num_of_samples] #torch.flatten(class_grads[num_of_samples:2*num_of_samples], start_dim=1)
                            layer_class_sim[c1,c2] = calculate_batch_stiffness(grads1, grads2)
                        else:
                            class1_indices = (target == c1)
                            class2_indices = (target == c2)
                            class1_grads = param.grad1[class1_indices].cuda()
                            #class1_grads -= param.grad
                            class2_grads = param.grad1[class2_indices].cuda()
                            #class2_grads -= param.grad
                            num_of_samples = min(len(class1_grads), len(class2_grads))
                            grads1 = class1_grads[:num_of_samples] #torch.flatten(class1_grads[:num_of_samples], start_dim=1)
                            grads2 = class2_grads[:num_of_samples] #torch.flatten(class2_grads[:num_of_samples], start_dim=1)
                            layer_class_sim[c1,c2] = layer_class_sim[c2,c1] = calculate_batch_stiffness(grads1, grads2)
                            
                        
                    except Exception as e:
                        if type(e) != AssertionError:
                            print(e)

                        stiffness_avg_dict.pop(name, None)
                        continue

            if name in stiffness_avg_dict:
                stiffness_avg_dict[name].update(layer_class_sim)

                avg_key = None        
                sub_name, indx, _ = name.split(".")
                if sub_name == "features": 
                    if int(indx) < 25:
                        avg_key = "early_conv_avg"
                    elif int(indx) >= 40:
                        avg_key = "late_conv_avg"

                if avg_key:
                    if avg_key not in stiffness_avg_dict:
                        stiffness_avg_dict[avg_key] = AverageMeter(torch.zeros([10, 10], dtype=torch.float32).cpu(), keep_val=False)

                    stiffness_avg_dict[avg_key].update(stiffness_avg_dict[name].get_average().cpu())

        stiffness_avg_dict["late_conv_avg"].softmax_average(threshold=0.1)
        stiffness_avg_dict["early_conv_avg"].softmax_average(threshold=0.1)


def calculate_batch_stiffness(grads1, grads2):
    '''
    grads1 = torch.cat((grads1, grads1), 0)
    grads2 = torch.cat((grads2, torch.flip(grads2, dims=[0,])), 0)
    sim = F.cosine_similarity(grads1, grads2, dim=1)
    sim[sim<0] = 0
    sim = torch.mean(sim, dim=0)
    '''

    grads1 = grads1.view(grads1.size(0),-1)
    grads1_c = grads1 - grads1.mean(dim=1).unsqueeze(1)
    grads1_n = grads1_c / (1e-8 + torch.sqrt(torch.sum(grads1_c**2, dim=1))).unsqueeze(1)

    grads2 = grads2.view(grads2.size(0),-1)
    grads2_c = grads2 - grads2.mean(dim=1).unsqueeze(1)
    grads2_n = grads2_c / (1e-8 + torch.sqrt(torch.sum(grads2_c**2, dim=1))).unsqueeze(1)
    
    R = grads1_n.matmul(grads2_n.transpose(1,0)).clamp(-1,1)
    R[R<0] = 0

    return torch.mean(R)

def plot_stiffness(epoch):
    global stiffness_avg_dict
    save_dir = 'figures/stiffness/{0}'.format(args.arch)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.clf()
    for k, STF in stiffness_avg_dict.items():
        _stiffness = STF.get_average().cpu().numpy()
        ax1 = sns.heatmap(_stiffness, vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
        plt.savefig(os.path.join(save_dir, "{1}_class_stiffness_{2}_EPOCH-{0}.png".format(epoch,args.arch, k)))
        plt.clf()

def update_AS(epoch):
    global AS_avg_dict

    for name, act_list in layer_activations.items():
        if name not in AS_avg_dict:
            AS_avg_dict[name] = torch.zeros([10, 10], dtype=torch.float32)

        for c1 in range(10):
            for c2 in range(c1, 10):
                c1_act_list = act_list[c1]
                c2_act_list = act_list[c2]

                c1_act = c1_act_list[:len(c1_act_list)//2]
                c2_act = c2_act_list[len(c2_act_list)//2:]

                stacked_c1 = np.vstack(c1_act)
                stacked_c2 = np.vstack(c2_act)

                
                AS_avg_dict[name][c1, c2] = _get_RV_avg_pool(stacked_c1, stacked_c2, activation_data_size).item()
                if c1 != c2:
                    AS_avg_dict[name][c2, c1] = AS_avg_dict[name][c1, c2]  

                if epoch == 0:
                    bl_name = name + "_BASELINE"
                    if bl_name not in AS_avg_dict:
                        AS_avg_dict[bl_name] = torch.zeros([10, 10], dtype=torch.float32)
                    AS_avg_dict[bl_name][c1, c2] = _get_RV_baseline(stacked_c1, stacked_c2, activation_data_size).item()
                    if c1 != c2:
                        AS_avg_dict[bl_name][c2, c1] = AS_avg_dict[bl_name][c1, c2]

            avg_key = None        
            sub_name, indx = name.split(".")
            if sub_name == "features": 
                if int(indx) < 25:
                    avg_key = "early_conv_avg"
                elif int(indx) >= 40:
                    avg_key = "late_conv_avg"

            if avg_key:
                if avg_key not in AS_avg_dict:
                    AS_avg_dict[avg_key] = AverageMeter(torch.zeros([10, 10], dtype=torch.float32).cpu(), keep_val=False)

                AS_avg_dict[avg_key].update(AS_avg_dict[name].cpu())

    AS_avg_dict["late_conv_avg"] = AS_avg_dict["late_conv_avg"].get_average()
    AS_avg_dict["early_conv_avg"] = AS_avg_dict["early_conv_avg"].get_average()

    layer_activations.clear()
    

def plot_AS(epoch):
    global stiffness_avg_dict
    save_dir = 'figures/class_similarity/{0}'.format(args.arch)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.clf()
    for k, AS in AS_avg_dict.items():
        if epoch != 0 and "BASELINE" in k:
            continue

        _sim = AS.cpu().numpy()
        ax1 = sns.heatmap(_sim, vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
        plt.savefig(os.path.join(save_dir, "{1}_actvtn_similarity_{2}_EPOCH-{0}.png".format(epoch,args.arch, k)))
        plt.clf()

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

if __name__ == '__main__':
    main()
