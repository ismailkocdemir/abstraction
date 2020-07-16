import os
import time
import argparse
import torch
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import weightwatcher as ww
import torchvision.models as torch_models
import models

parser = argparse.ArgumentParser(description='Weight Analysis')

parser.add_argument('--architecture-1', '--a-1', default='resnet18k_4',
                help='architecture of the first model (default: resnet18k_4)')
parser.add_argument('--architecture-2', '--a-2', default='resnet18k_4',
                help='architecture of the second model (default: resnet18k_4)')
parser.add_argument('--epoch-1', '--e-1', default=0, type=int,
                 help='epoch of the first model\'s saved checkpoint (default: 0)')
parser.add_argument('--epoch-2', '--e-2', default=0, type=int,
                 help='epoch of the second model\'s saved checkpoint (default: 0)')

def ww_demo():
    model = torch_models.vgg11(pretrained=True)
    watcher = ww.WeightWatcher(model=model)
    results = watcher.analyze()

    summary = watcher.get_summary()

def plot_helper(arch1, arch2, epoch1, epoch2, data_arch1, data_arch2):
    plt.clf()
    plt.figure()
    big_list_x = []
    big_list_y = []
    for idx, y in enumerate(data_arch1):
        big_list_x += (idx*np.ones((len(y),))).tolist()
        big_list_y += y
    
    plt.scatter(big_list_x, big_list_y, label="{} e-{}".format(arch1, epoch1), c='r')

    big_list_x = []
    big_list_y = []
    for idx, y in enumerate(data_arch2):
        big_list_x += (idx*np.ones((len(y),))).tolist()
        big_list_y += y
        
    plt.scatter(big_list_x, big_list_y, label="{} e-{}".format(arch2, epoch2), c='b')
    plt.legend()
    plt.xlabel("Layer ID")
    plt.ylabel("Alpha")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig("figures/lognorm/alpha_{}-e{}_{}-e{}_{}".format(arch1, epoch1, arch2, epoch2, timestamp))


def get_lognorm_per_layer(arch, epoch):
    model = models.__dict__[arch]()
    
    #if "vgg" in arch.lower():
    #    model.features = torch.nn.DataParallel(model.features)

    print("=> loading checkpoint '{}'".format(epoch))
    
    cp_path = os.path.join("checkpoints", arch, "checkpoint_{0}.tar".format(epoch))
    checkpoint = torch.load(cp_path)
    best_prec1 = checkpoint['best_prec1']
    print("Accuracy in Val. Set: {0}".format(best_prec1))
    model.load_state_dict(checkpoint['state_dict'])
    model.cpu()

    watcher = ww.WeightWatcher(model=model)
    results = watcher.analyze(min_size=50, alphas=True)

    data_per_layer = [[] for item in results]
    
    for layer_id, result in results.items():
        for slice_id, summary in result.items():
            if not str(slice_id).isdigit() or "alpha_weighted" not in summary:
                continue
            
            #lognorm = summary["lognorm"]
            alpha_weighted = summary["alpha_weighted"]
            data_per_layer[layer_id].append(alpha_weighted)
            

            print("Layer {}, Slice {}: Alpha: {}".format(layer_id, slice_id, alpha_weighted))

    return data_per_layer

def analyze_weights(arch_1, arch_2, epoch_1, epoch_2):
    arch1_data_per_layer = get_lognorm_per_layer(arch_1, epoch_1)
    arch2_data_per_layer = get_lognorm_per_layer(arch_2, epoch_2)

    plot_helper(arch_1, arch_2, epoch_1, epoch_2, arch1_data_per_layer, arch2_data_per_layer)


if  __name__ == "__main__":
    args = parser.parse_args()
    analyze_weights(args.architecture_1, args.architecture_2, args.epoch_1, args.epoch_2)