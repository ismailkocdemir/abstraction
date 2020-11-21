import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
import sys

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class JacobianRegularizerCrossEntropyLoss:
    def __init__(self, reduction="mean", alpha=5e-2, _lambda=0.95):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.is_cuda = False
        self.alpha = alpha
        self._lambda = _lambda

    def __call__(self, output, target, jacobian=None):
        ce_loss = self.cross_entropy(output, target)

        if jacobian == None:
            return ce_loss

        jacobian_loss = 0.0
        for idx, out in enumerate(jacobian):
            penalty_ratio = self._lambda ** (len(jacobian)-idx-1)

            loss = (jacobian[idx] ** 2).sum(dim=[i for i in range(1, len(jacobian[idx].shape))]).mean()
            jacobian_loss += penalty_ratio * loss

        return ce_loss + 0.5 * self.alpha * jacobian_loss

    def cuda(self,):
        self.cross_entropy.cuda()
        self.is_cuda = True
        return self

    def cpu(self,):
        self.cross_entropy.cpu()
        self.is_cuda = False
        return self

class TransformSensitivityCrossEntropyLoss:
    def __init__(self, reduction="mean", alpha=5e-2, sim_loss="cosine", _lambda=0.95, block=False, strength="decrease"):

        assert sim_loss.lower() in ["mse", "cosine"], "{} is an undefined metric. Choose from ('mse', 'cosine')".format(sim_loss.lower())
        assert strength.lower() in ["increase", "decrease", "constant", "increase_exp"]

        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.is_cuda = False
        self.alpha = alpha
        self._lambda = _lambda
        self.block = block
        self.sim_loss = sim_loss.lower()
        self.strength = strength.lower()

    def __call__(self, output, target, noisy_output=None):
        ce_loss = self.cross_entropy(output[-1], target)

        if noisy_output == None:
            return ce_loss

        noise_sensitivity_loss = 0.0
        for idx, out in enumerate(output):
            if (idx%5 or idx==0) and self.block:
                continue
            if self.strength == "increase":
                penalty_ratio = (idx//5 + 1) / len(output)
            elif self.strength == "decrease":
                penalty_ratio = 1 / (idx//5 + 1)
            elif self.strength == "constant":
                penalty_ratio = 1
            elif self.strength == "increase_exp":
                penalty_ratio = self._lambda ** (len(output)-idx-1)

            if self.sim_loss == "mse":
                sum_dims = [i+1 for i in range(len(out.shape)-1)]
                loss = F.mse_loss(out, noisy_output[idx] ,reduction='none').sum(dim=sum_dims)
                loss /= torch.norm(out.flatten(start_dim=1), dim=1)**2
                noise_sensitivity_loss += penalty_ratio * loss.mean()
            elif self.sim_loss == "cosine":
                cosine_target = torch.ones((out.size(0),)).to(out.device)
                if self.is_cuda:
                    cosine_target.cuda()
                noise_sensitivity_loss += penalty_ratio * F.cosine_embedding_loss(out.view((out.shape[0], -1)),
                                                                noisy_output[idx].view((out.shape[0], -1)),
                                                                cosine_target
                                                                )
        return ce_loss + self.alpha * noise_sensitivity_loss

    def cuda(self,):
        self.cross_entropy.cuda()
        self.is_cuda = True
        return self

    def cpu(self,):
        self.cross_entropy.cpu()
        self.is_cuda = False
        return self

class AdaptiveDecayCrossEntropyLoss:
    def __init__(self, reduction="mean", alpha=1e-3, _lambda = 0.95, strength="constant", blocks=4):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.is_cuda = False
        self.strength = strength
        self.alpha = alpha
        self._lambda = _lambda
        self.blocks = blocks

    def __call__(self, output, target, model=None):
        ce_loss = self.cross_entropy(output, target)

        if model == None:
            return ce_loss

        norm_loss = 0.0
        total_size = sum([1 if "bias" not in n else 0 for (n,p) in model.named_parameters()])
        prev_jacob = None
        idx = 0
        for n, p in model.named_parameters():
            # if self._lambda != 0:

            if self.strength == "increase":
                penalty_ratio = (idx//self.blocks + 1) / total_size
            elif self.strength == "decrease":
                penalty_ratio = 1 / (idx//self.blocks + 1)
            elif self.strength == "constant":
                penalty_ratio = 1
            elif self.strength == "increase_exp":
                penalty_ratio = self._lambda ** (total_size-idx-1)

            if "bias" in n:
                norm_loss += penalty_ratio * (p**2).sum()
                idx += 1
            else:
                if prev_jacob != None:
                    if len(p.shape) > 2:
                        prev_jacob = torch.matmul(p.mean(dim=(2,3)), prev_jacob)
                    else:
                        prev_jacob = torch.matmul(p, prev_jacob)
                else:
                    prev_jacob = p.mean(dim=(2,3))
                norm_loss += penalty_ratio * (prev_jacob**2).sum()
            # If lambda is 0, adaptive decay defaults to standart weight decay
            # else:
            # norm_loss += (p**2).sum()
        return ce_loss + self.alpha * 0.5 * norm_loss

    def cuda(self,):
        self.cross_entropy.cuda()
        self.is_cuda = True
        return self

    def cpu(self,):
        self.cross_entropy.cpu()
        self.is_cuda = False
        return self

def regularize_spectral_norm(model, v_dict, alpha, _lambda, strength, blocks=4):
    total_size = len(v_dict) #sum([1 if ("bias" not in n and ".bn" not in n) else 0 for (n,p) in model.named_parameters() ])
    idx = 0
    for n, p in model.named_parameters():
        if p.grad == None:
            continue
        flag = False
        if n in v_dict:
            penalty, new_v = power_iteration(p, v_dict[n])
            v_dict[n] = new_v / torch.norm(new_v)
            flag = True
        #If param is not an instance of nn.Linear or nn.Conv2d, apply regular decay
        else:
            # TODO: Regular weight decay has a lower ratio. Adjust accordingly.
            penalty = 5e-2 * p.data

        if strength == "increase":
            penalty_ratio = (idx//blocks + 1) / total_size
        elif strength == "decrease":
            penalty_ratio = 1 / (idx//blocks + 1)
        elif strength == "constant":
            penalty_ratio = 1
        elif strength == "increase_exp":
            penalty_ratio = _lambda ** (total_size-idx-1)

        p.grad += alpha * penalty * penalty_ratio
        if flag:
            idx += 1

def power_iteration(param, v):
    param_copy = param.clone().detach()
    with torch.no_grad():
        if len(param.shape) > 2:
            param_copy = param_copy.view((param_copy.shape[0],-1))
        #print(param.size(), param_copy.size(), v.size())
        u = torch.mv(param_copy, v)
        v = torch.mv(torch.transpose(param_copy, 0, 1), u)
        eig = torch.norm(u) / torch.norm(v)
        uv_t = torch.matmul(u.unsqueeze(1), v.unsqueeze(1).t())
        if len(param.shape) > 2:
            uv_t = uv_t.view(param.size())
    return eig * uv_t, v

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        std_in_ratio = torch.norm(tensor, dim=(1,2), keepdim=True) * self.std
        return tensor + torch.randn(tensor.size()).to(std_in_ratio.device) * std_in_ratio  + self.mean
        #std_in_ratio = torch.norm(tensor) * self.std
        #return tensor + torch.randn(tensor.size()) * std_in_ratio  + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, initial=0, keep_val=True):
        self.keep_val = keep_val
        self.reset(initial)

    def reset(self, initial = 0):
        if self.keep_val:
            self.val = initial
        else:
            self.val = 0
        self.avg = initial
        self.count = 0

    def update(self, val, n=1):
        if self.keep_val:
            self.val = val

        self.avg = self.avg*self.count + val*n
        self.count += n
        self.avg = self.avg/self.count

    def get_average(self,):
        return self.avg

    def softmax_average(self, threshold=0.1):
        self.avg[self.avg<0.1] = 0.0
        self.avg[self.avg>0.1] = 1.0
        self.avg = F.softmax(self.avg, dim=1)

def get_train_loader_CIFAR10(batch_size, workers, label_noise=False, resize=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    _transforms  = [transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(resize if resize else 32, 4),
        transforms.ToTensor(),
        normalize
    ]

    if resize:
        _transforms.insert(0, transforms.Resize(resize))

    _transforms = transforms.Compose(_transforms)
    train_set = datasets.CIFAR10(root='./data', train=True, transform=_transforms, download=True)

    if label_noise > 0:
        set_size = len(train_set.targets)
        percentage = label_noise + label_noise/9.0
        random_indexes = np.random.choice(set_size, int(set_size*percentage/100.0), replace=False)
        random_labels = np.random.choice(10, len(random_indexes), replace=True)
        i = 0
        for idx in random_indexes:
            train_set.targets[idx] = random_labels[i]
            i += 1

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)
    return train_loader

def get_val_loader_CIFAR10(batch_size, workers, resize=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    _transforms_val = [transforms.ToTensor(),
                        normalize
                    ]

    if resize:
        _transforms_val.insert(0, transforms.Resize(resize))
    #if noise_sigma:
    #    _transforms_val.append(AddGaussianNoise(0, noise_sigma))

    _transforms_val = transforms.Compose(_transforms_val)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=_transforms_val, download=True),
                        batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=True)
    return val_loader

def get_train_loader_CIFAR100(batch_size, workers, label_noise=0, resize=None):
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    transform_train = [
        #transforms.ToPILImage(),
        transforms.RandomCrop(resize if resize else 32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]
    if resize:
        transform_train.insert(0, transforms.Resize(resize))
    transform_train = transforms.Compose(transform_train)
    train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

    if label_noise > 0:
        set_size = len(train_set.targets)
        percentage = label_noise + label_noise/9.0
        random_indexes = np.random.choice(set_size, int(set_size*percentage/100.0), replace=False)
        random_labels = np.random.choice(10, len(random_indexes), replace=True)
        i = 0
        for idx in random_indexes:
            train_set.targets[idx] = random_labels[i]
            i += 1

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)
    return train_loader

def get_val_loader_CIFAR100(batch_size, workers, resize=None):
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

    transform_test = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    if resize:
        transform_test.insert(0, transforms.Resize(resize))

    transform_test = transforms.Compose(transform_test)
    val_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(
        val_set, shuffle=False, num_workers=workers, batch_size=batch_size)
    return val_loader

def add_gaussian_noise(tensor, mean = 0.0, std=0.1):
    std_in_ratio = torch.norm(tensor, dim=(2,3), keepdim=True) * std
    return tensor + torch.randn(tensor.size()).to(std_in_ratio.device) * std_in_ratio  + mean

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_dictionary_to_file(filepath, dictnry):
    """Dumps the content of a dictionary into pickle file """
    with open(filepath, 'wb') as handle:
        pickle.dump(dictnry, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dictionary_from_file(filepath):
    """Reads the pickle file into a dictionary"""
    with open(filepath, 'wb') as handle:
        dictnry = pickle.load(handle)
        return dictnry

def convert_to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

def get_beta(avg_gen_error):
    beta = avg_gen_error if avg_gen_error >= 0.5 else 0.5
    beta = beta if beta <= 1.0 else 1.0
    beta = -1*np.log10(1.1-beta)
    return 1-beta

def threshold_gradient_stiffness(similarity, threshold=0.05, steepness=0.25):
    similarity = similarity.cpu().numpy()
    if similarity < threshold + 1e-5:
        return 1e-5
    return np.exp(((similarity-threshold)**2)/steepness)

"""
class AdaptiveDecayCrossEntropyLoss:
    def __init__(self, reduction="mean", alpha=1e-3, _lambda = 0.5):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.is_cuda = False
        self.alpha = alpha
        self._lambda = _lambda

    def __call__(self, output, target, model=None):
        ce_loss = self.cross_entropy(output, target)

        if model == None:
            return ce_loss

        norm_loss = 0.0
        idx = sum([1 if hasattr(p, "weight") else 0 for (n,p) in model.named_parameters() ])
        prev_jacob = None
        norm_product = 1.0
        for n, p in model.named_parameters():
            if self._lambda != 0 and "bias" not in n:
                adaptive_ratio = self._lambda ** (idx)
                if prev_jacob != None:
                    if len(p.shape) > 2:
                        prev_jacob = F.conv2d(prev_jacob.unsqueeze(0), p, padding=1).squeeze(0)
                    else:
                        if len(prev_jacob.shape) > 2:
                            pool_size = int((prev_jacob.numel() / p.size(1))**0.5)
                            prev_jacob = torch.matmul(p, F.max_pool2d(prev_jacob.unsqueeze(0), pool_size).squeeze(0).flatten(start_dim=1))
                        else:
                            prev_jacob = torch.matmul(p, prev_jacob)
                else:
                    prev_jacob = F.conv2d(torch.ones((1,3,32,32)).to(p.device), p, padding=1).squeeze(0)
                norm_loss += adaptive_ratio * ((prev_jacob*prev_jacob).sum()**0.5)
                idx -= 1
            # If lambda is 0, adaptive decay defaults to standart weight decay
            else:
                norm_loss += (p*p).sum()

        return ce_loss + self.alpha * norm_loss

    def cuda(self,):
        self.cross_entropy.cuda()
        self.is_cuda = True
        return self

    def cpu(self,):
        self.cross_entropy.cpu()
        self.is_cuda = False
        return self
"""

"""
class JacobianRegularizerCrossEntropyLoss:
    def __init__(self, reduction="mean", alpha=5e-2):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.alpha = alpha

    def __call__(self, output, target, model=None, input=None):
        loss = self.cross_entropy(output, target)
        if model == None and input == None:
            return loss
        dL_dx = torch.autograd.grad(loss, input, create_graph=True)[0]
        return loss + 0.5 * self.alpha * ((dL_dx)**2).sum()

    def cuda(self,):
        self.cross_entropy.cuda()
        return self

    def cpu(self,):
        self.cross_entropy.cpu()
        return self
"""
