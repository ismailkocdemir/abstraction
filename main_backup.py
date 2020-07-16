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
from representation import _get_pwcca_avg_pool, _get_pwcca_baseline, _get_RV_avg_pool, _get_RV_baseline

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


stiffness_avg_dict = defaultdict()
interpol_epoch = 10
stiffness_data_size = 1024

AS_updated = []
first_update = False
is_best = False
gen_error = []

#layer_activations = defaultdict()
#layer_gradients = defaultdict()

best_prec1 = 0
prev_prec = 0

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
parser.add_argument('--stiffness', dest='stiffness', action='store_true',
                    help='Collect stiffness values for gradients')
parser.add_argument('-r', '--refer-initial', dest='refer_init', action='store_true',
                    help='Collect activations that compare to their checkpoint versions OR \
                    cosine distances that refer to initial random weights')
parser.add_argument('--assert-lr', dest="assert_lr", action='store_true',
                    help='Assert larger initial learning rate when resuming, as opposed to \
                    checkpoint adjustment.')
parser.add_argument('--constant-lr', dest="constant_lr", action='store_true',
                    help='Constant learning rate.')



def main():
    global args, best_prec1, gen_error, is_best, layer_activations, activations, prev_prec, interpol_epoch
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = models.__dict__[args.arch]()

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
    criterion = nn.KLDivLoss(reduction='batchmean')
    #criterion = nn.CrossEntropyLoss(reduction='mean')
    
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
    gen_error = [1.0]
        
    for epoch in range(args.start_epoch, args.epochs):
        if not args.constant_lr:
            adjust_learning_rate(optimizer, epoch, AS_updated)

        # train for one epoch
        if args.collect_activations:
            layer_activations.clear()
            activations.clear()

        prec1_train, loss_train = train(train_loader, model, criterion, optimizer, epoch)        

        # evaluate on validation set
        prec1, loss_val = validate(val_loader, model, criterion, epoch)
        
        curr_gen_error_ratio = np.abs(loss_train/loss_val)
        gen_error.append(curr_gen_error_ratio)
        
        if args.collect_activations:
            plot_AS(epoch, curr_gen_error_ratio, prec1_train, prec1)
        if args.stiffness and epoch >= interpol_epoch  and epoch%10 == 0:
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
    global AS_val, interpol_epoch, stiffness_data_size

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    assert interpol_epoch % 10 == 0, "Interpolation epoch must be a multiple of 10"

    if args.stiffness and epoch >= interpol_epoch and epoch%10 == 0:
        enable_hooks()
        add_hooks(model)

    # switch to train mode
    model.train()

    end = time.time()
    
    if epoch <= interpol_epoch:
        beta = 1.0
    else:
        beta = 0.5 # 0.5 + 1.0/(epoch//3) #get_beta(avg_gen_error_ratio)
    print("BETA value: {}".format(beta))

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

        if args.stiffness and epoch >= interpol_epoch and args.batch_size*(i+1) >= stiffness_data_size  and epoch%10 == 0:
            disable_hooks()
            remove_hooks(model)

        output = model(input)
        
        ## Required conversions for KL loss.
        output_logp = F.log_softmax(output, dim=1)
        one_hot_target = convert_to_one_hot(target, 10).cuda().float()
        if epoch > interpol_epoch:
            sim_dist = torch.index_select(stiffness_avg_dict["late_conv_avg"].get_average(), 0, target.cpu()).cuda().float()
            target_dist = (beta*one_hot_target + (1-beta)*sim_dist)
            if i==0:
                print(sim_dist[:3])
                print(target_dist[:3])
            loss = criterion(output_logp.cuda(async=True), target_dist.cuda(async=True))
        else:
            loss = criterion(output_logp.cuda(async=True), one_hot_target.cuda(async=True))

        ## For CrossEntropy Loss
        #loss = criterion(output, target)

        # compute gradient
        loss.backward()

        if args.stiffness and epoch >= interpol_epoch and args.batch_size*(i+1) < stiffness_data_size and epoch%10 == 0:
            update_stiffness(model, target)

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
    global activations, interpol_epoch
    global AS_updated
    global update_patience
    global counter
    global gen_error    
    global first_update
    global is_best
    global layer_activations
    global AS_val
    #global layer_activations_prev

    assert interpol_epoch % 5 == 0, "Interpolation epoch must be a factor of 5"

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    should_update = True

    if not args.evaluate and args.collect_activations:
    if args.collect_activations:
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
                        elif sub_name == "classifier" and int(indx) in [1,4,6]:
                            hooks.append(m.register_forward_hook(partial(get_activation, name)))

    if epoch<=interpol_epoch:
        beta = 1.0
    else:
        beta = 0.5 #0.5 + 1.0/(epoch//3) #get_beta(avg_gen_error_ratio)
    
    if not args.evaluate and (args.collect_activations or (args.stiffness and epoch==interpol_epoch)):
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
            
            ## Required conversions for KL loss.
            output_logp = F.log_softmax(output, dim=1)
            one_hot_target = convert_to_one_hot(target, 10).cuda().float()
            if epoch > interpol_epoch:
                sim_dist = torch.index_select(stiffness_avg_dict["late_conv_avg"].get_average(), 0, target.cpu()).cuda().float()
                target_dist = (beta*one_hot_target + (1-beta)*sim_dist)
                loss = criterion(output_logp.cuda(async=True), target_dist.cuda(async=True))
            else:
                loss = criterion(output_logp.cuda(async=True), one_hot_target.cuda(async=True))

        if args.collect_activations:
            if args.batch_size*(i+1) > 4096: 
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
                        v_np = v.clone().detach().cpu().numpy()
                        for c in range(10):
                            if c not in layer_activations.keys():
                                 layer_activations[c] = defaultdict(list)
                            class_indexes = (target.cpu().numpy() == c)
                            layer_activations[c][k].append(v_np[class_indexes])

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        #accumulate class activations
        if not args.evaluate and (args.collect_activations or (args.stiffness and epoch==interpol_epoch)):
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
                #else:
                #    print("zero size")


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

    if not args.evaluate and ((args.collect_activations or args.stiffness) and epoch==interpol_epoch):
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

def adjust_learning_rate(optimizer, epoch, AS_updated):
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

def update_stiffness(model, target):
    """ Compute the class similarity matices from the gradient stiffness values """
    global stiffness_avg_dict
    with torch.no_grad():
        compute_grad1(model, 'mean')

        for name, param in model.named_parameters():
            if 'bias' in name:
                continue

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
    grads1 = torch.cat((grads1, grads1), 0)
    grads2 = torch.cat((grads2, torch.flip(grads2, dims=[0,])), 0)
    sim = F.cosine_similarity(grads1, grads2, dim=1)
    sim[sim<0] = 0
    sim = torch.mean(sim, dim=0)

    if grad1.dim() == 4:
        x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
        else:
            x = x.view(x.size(0),-1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc**2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1,0)).clamp(-1,1)
    return sim

def plot_stiffness(epoch):
    global stiffness_avg_dict
    save_dir = 'figures/stiffness/{0}'.format(args.arch)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.clf()
    for k, STF in stiffness_avg_dict.items():
        _stiffness = STF.get_average().cpu().numpy()
        ax1 = sns.heatmap(_stiffness, vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
        plt.savefig(os.path.join(save_dir, "{1}_class_sim_{2}_EPOCH-{0}.png".format(epoch,args.arch, k)))
        plt.clf()


def update_AS(epoch):
    global layer_activations
    global AS_final, AS_fc, AS_highest, AS_high, AS_mid_high, AS_mid, AS_mid_low
    global baseline_AS_fc, baseline_AS_highest, baseline_AS_high, baseline_AS_mid_high, baseline_AS_mid, baseline_AS_mid_low

    for c1 in range(10):
        for c2 in range(c1, 10):
            c1_act_list = layer_activations[c1]
            c2_act_list = layer_activations[c2]
            similarity_final = 0.0
            similarity_fc = 0.0
            similarity_highest = 0.0
            similarity_high = 0.0
            similarity_mid_high = 0.0
            similarity_mid = 0.0
            similarity_mid_low = 0.0

            baseline_similarity_fc = 0.0
            baseline_similarity_highest = 0.0
            baseline_similarity_high = 0.0
            baseline_similarity_mid_high = 0.0
            baseline_similarity_mid = 0.0
            baseline_similarity_mid_low = 0.0

            for k, c1_act_o in c1_act_list.items():
            
                c1_act = c1_act_o[:len(c1_act_o)//2]
                c2_act = c2_act_list[k][len(c2_act_list[k])//2:]

                splt = k.split(".")
                sub_name, indx = splt
                stacked_c1 = np.vstack(c1_act)
                stacked_c2 = np.vstack(c2_act)

                #if sub_name == "classifier" and int(indx) == 6:
                #    current_sim = _get_pwcca_avg_pool(stacked_c1, stacked_c2, 512)
                #else:
                current_sim = _get_RV_avg_pool(stacked_c1, stacked_c2, 512)

                if epoch == 0:
                    current_sim_baseline = _get_RV_baseline(stacked_c1, stacked_c2, 512)

                if sub_name == "features":
                    if int(indx) in [20,23]:#[3,7,10]:
                        similarity_mid_low += current_sim
                        if epoch == 0:
                            baseline_similarity_mid_low = current_sim_baseline
                    elif int(indx) in [27,30]:#[14,17,20,23]:
                        similarity_mid += current_sim
                        if epoch == 0:
                            baseline_similarity_mid = current_sim_baseline
                    elif int(indx) in [33,36]:#[27,30,33,36]:
                        similarity_mid_high += current_sim
                        if epoch == 0:
                            baseline_similarity_mid_high = current_sim_baseline
                    elif int(indx) in [40,43]:
                        similarity_high += current_sim
                        if epoch == 0:
                            baseline_similarity_high = current_sim_baseline
                    elif int(indx) in [46,49]:
                        similarity_highest += current_sim
                        if epoch == 0:
                            baseline_similarity_highest = current_sim_baseline
                else:
                    if int(indx) in [1,4]:
                        similarity_fc += current_sim
                    else:
                        similarity_final += current_sim
                    if epoch == 0:
                        baseline_similarity_fc = current_sim_baseline
        
            AS_final[c1,c2] = similarity_final
            AS_fc[c1,c2] = similarity_fc/2
            AS_highest[c1,c2] = similarity_highest/2
            AS_high[c1,c2] = similarity_high/2
            AS_mid_high[c1,c2] = similarity_mid_high/2
            AS_mid[c1,c2] = similarity_mid/2
            AS_mid_low[c1,c2] = similarity_mid_low/2

            if epoch == 0:
                baseline_AS_fc[c1,c2] = baseline_similarity_fc/2
                baseline_AS_highest[c1,c2] = baseline_similarity_highest/2
                baseline_AS_high[c1,c2] = baseline_similarity_high/2
                baseline_AS_mid_high[c1,c2] = baseline_similarity_mid_high/2
                baseline_AS_mid[c1,c2] = baseline_similarity_mid/2
                baseline_AS_mid_low[c1,c2] = baseline_similarity_mid_low/2
        
    for c1 in range(10):
        for c2 in range(c1):
            AS_final[c1,c2] = AS_final[c2,c1]
            AS_fc[c1,c2] = AS_fc[c2,c1]
            AS_highest[c1,c2] = AS_highest[c2,c1]
            AS_high[c1,c2] = AS_high[c2,c1]
            AS_mid_high[c1,c2] = AS_mid_high[c2,c1]
            AS_mid[c1,c2] = AS_mid[c2,c1]
            AS_mid_low[c1,c2] = AS_mid_low[c2,c1]

            if epoch == 0:
                baseline_AS_fc[c1,c2] = baseline_AS_fc[c2,c1]
                baseline_AS_highest[c1,c2] = baseline_AS_highest[c2,c1]
                baseline_AS_high[c1,c2] = baseline_AS_high[c2,c1]
                baseline_AS_mid_high[c1,c2] = baseline_AS_mid_high[c2,c1]
                baseline_AS_mid[c1,c2] = baseline_AS_mid[c2,c1]
                baseline_AS_mid_low[c1,c2] = baseline_AS_mid_low[c2,c1]

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
    plt.savefig("figures/class_sim/{4}/{4}_class_sim_MID_LOW_EPOCH-{0}_gen-error-{1:.2f}__{2:.1f}-{3:.1f}.png".format(epoch, curr_gen_error_ratio, acc_train, acc_val, args.arch))
    plt.clf()
    ax2 = sns.heatmap(AS_mid.cpu().numpy(), vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
    plt.savefig("figures/class_sim/{4}/{4}_class_sim_MID_EPOCH-{0}_gen-error-{1:.2f}__{2:.1f}-{3:.1f}.png".format(epoch, curr_gen_error_ratio, acc_train, acc_val, args.arch))
    plt.clf()
    ax3 = sns.heatmap(AS_mid_high.cpu().numpy(), vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
    plt.savefig("figures/class_sim/{4}/{4}_class_sim_MID_HIGH_EPOCH-{0}_gen-error-{1:.2f}__{2:.1f}-{3:.1f}.png".format(epoch, curr_gen_error_ratio, acc_train, acc_val, args.arch))
    plt.clf()
    ax4 = sns.heatmap(AS_high.cpu().numpy(), vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
    plt.savefig("figures/class_sim/{4}/{4}_class_sim_HIGH_EPOCH-{0}_gen-error-{1:.2f}__{2:.1f}-{3:.1f}.png".format(epoch, curr_gen_error_ratio, acc_train, acc_val, args.arch))
    plt.clf()
    ax5 = sns.heatmap(AS_highest.cpu().numpy(), vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
    plt.savefig("figures/class_sim/{4}/{4}_class_sim_HIGHEST_EPOCH-{0}_gen-error-{1:.2f}__{2:.1f}-{3:.1f}.png".format(epoch, curr_gen_error_ratio, acc_train, acc_val, args.arch))
    plt.clf()
    ax6 = sns.heatmap(AS_fc.cpu().numpy(), vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
    plt.savefig("figures/class_sim/{4}/{4}_class_sim_FC_EPOCH-{0}_gen-error-{1:.2f}__{2:.1f}-{3:.1f}.png".format(epoch, curr_gen_error_ratio, acc_train, acc_val, args.arch))
    plt.clf()
    ax0 = sns.heatmap(AS_final.cpu().numpy(), vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
    plt.savefig("figures/class_sim/{4}/{4}_class_sim_FINAL_EPOCH-{0}_gen-error-{1:.2f}__{2:.1f}-{3:.1f}.png".format(epoch, curr_gen_error_ratio, acc_train, acc_val, args.arch))

    if epoch == 0:
        plt.clf()
        ax7 = sns.heatmap(baseline_AS_mid_low.cpu().numpy(), vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
        plt.savefig("figures/class_sim/{0}/{0}_class_sim_MID_LOW_BASELINE.png".format(args.arch))
        plt.clf()
        ax8 = sns.heatmap(baseline_AS_mid.cpu().numpy(), vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
        plt.savefig("figures/class_sim/{0}/{0}_class_sim_MID_BASELINE.png".format(args.arch))
        plt.clf()
        ax9 = sns.heatmap(baseline_AS_mid_high.cpu().numpy(), vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
        plt.savefig("figures/class_sim/{0}/{0}_class_sim_MID_HIGH_BASELINE.png".format(args.arch))
        plt.clf()
        ax10 = sns.heatmap(baseline_AS_high.cpu().numpy(), vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
        plt.savefig("figures/class_sim/{0}/{0}_class_sim_HIGH_BASELINE.png".format(args.arch))
        plt.clf()
        ax11 = sns.heatmap(baseline_AS_highest.cpu().numpy(), vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
        plt.savefig("figures/class_sim/{0}/{0}_class_sim_HIGHEST_BASELINE.png".format(args.arch))
        plt.clf()
        ax12 = sns.heatmap(baseline_AS_fc.cpu().numpy(), vmin=0.0, vmax=1.0, square=True, annot=True, fmt=".2f")
        plt.savefig("figures/class_sim/{0}/{0}_class_sim_FC_BASELINE.png".format(args.arch))



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
