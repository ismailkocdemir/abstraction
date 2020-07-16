import os
import sys
import traceback
import argparse
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
import pandas
import gzip
from collections import defaultdict
import time
from scipy.stats import entropy as entropy_scipy

from math import log, e

sys.path.append("svcca")
import cca_core
import pwcca
import models
from utils import *
from autograd_hacks import is_supported

parser = argparse.ArgumentParser(description='Representational Similarity from Activations')

parser.add_argument('--run-mode', '--rm', default='in-net',
                help='Running mode: {in-net, between-nets, abstraction}')
parser.add_argument('--method', '-m', default='RV',
                help='Correlation method to use: mean SVCCA OR PWCCA (default: RV)')
parser.add_argument('--metric', default='kl-data', help='Abstractness measure metric (choices: kl-data, kl-neuron, entropy-sv)')

parser.add_argument('--architecture', '-a', default='resnet18k_16',
                help='architecture (default: cnn5)')
parser.add_argument('--architecture2', '--a2', default='resnet18k_16',
                help='Second architecture (default: resnet18k_16)')
parser.add_argument('--reference-layer-idx', '-r', default=-1, type=int,
                metavar='N', help='Similarity is calculated against this layer for all layers (default: -1)')
parser.add_argument('--num-datapoints', '-n', default=500, type=int,
                metavar='D', help='Number of datapoints used OR Dimension of the activation space (default: 500)')
parser.add_argument('--min-epoch', default=0, type=int,
                 help='first epoch of the model for the activations (default: 0)')
parser.add_argument('--max-epoch', '-e', default=-1, type=int,
                 help='last epoch of the model for the activations (default: -1)')
parser.add_argument('--epoch-a1', "--e1", default=-1, type=int,
                 help='epoch of the first model for the activations (default: -1)')
parser.add_argument('--epoch-a2', '--e2', default=-1, type=int,
                 help='epoch of the second model for the activations (default: -1)')
parser.add_argument('--step', '-s', default=1, type=int,
                 help='step between epochs (default: 1)')


def _plot_helper(arr_x, arr_y, baseline_y,  xlabel, ylabel, save_path="svcca_default.png"):
    plt.plot(arr_x, arr_y, label="Layers")
    plt.plot(arr_x, baseline_y, label="Baseline")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')    
    plt.grid()
    plt.savefig(save_path)

def _plot_helper_per_epoch(arr_x, values, ylabel, baseline, epochs, save_path="plot_default.png", in_net=True):

    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.grid()
    plt.subplots_adjust(bottom=0.25)

    if baseline:
        baseline = baseline[1:]
    arr_x = arr_x[:-1]
    values = [item[1:] for item in values]
    
    if epochs == None:
        plt.plot(arr_x, values[0], color='b', marker="o", linestyle="-")
        
        splt = save_path.split('/')
        splt.insert(-1, 'progress')
        save_path_new = os.path.join(*splt)

        plt.savefig(save_path_new)
        plt.clf()
        return

    c = np.array([item for item in epochs if item!='baseline'])

    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.inferno)
    cmap.set_array([])
    
    if baseline:
        plt.plot(arr_x, baseline,  color='k', linestyle='--', label='baseline')
    for idx, epoch in enumerate(epochs):
        if epoch == 'baseline':
            plt.plot(arr_x, values[idx],  color='k', linestyle='--', label='baseline')
        else:
            plt.plot(arr_x, values[idx], c=cmap.to_rgba(epoch))

    cbar = plt.colorbar(cmap, ticks=[c[0], c[-1]])
    cbar.set_label("Epoch")
    plt.savefig(save_path)
    plt.clf()

def demo():
    # Toy Example of CCA in action

    # assume X_fake has 100 neurons and we have their activations on 1000 datapoints
    A_fake = np.random.randn(100, 2000)
    # Y_fake has 50 neurons with activations on the same 1000 datapoints
    # Note X and Y do *not* have to have the same number of neurons
    B_fake = np.random.randn(50, 2000)

    # computing CCA simliarty between X_fake, Y_fake
    # We expect similarity should be very low, because the fake activations are not correlated
    results = cca_core.get_cca_similarity(A_fake, B_fake, verbose=True)

    print("Returned Information:")
    print(results.keys())
    print("MEAN SVCCA:", np.mean(results["cca_coef1"]))

    _plot_helper(list(range(len(results["cca_coef1"]))), results["cca_coef1"],  
                        results["cca_coef1"],"CCA coef idx", "CCA coef value", "cca_demo.png")

def _gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q

def KL_along_neuron_axis(values, verbose=False):
    """ Computes entropy of a set of values"""
    uniform = np.ones_like(values)
    uniform /= uniform.shape[0]
    
    if verbose:
        print(values)

    values_stable = (values - values.min()) + 1e-10
    #hist = np.histogram(values_stable, bins='auto', density=True)
    #data = hist[0]/hist[0].sum()
    #ent = -(data*np.ma.log(np.abs(data))).sum()

    ent = entropy_scipy(values_stable, uniform)

    return ent

def KL_along_data_axis(values, verbose=False):
    uniform = np.ones_like(values)
    uniform /= uniform.shape[0]
    
    if verbose:
        print(values)

    values_stable = (values - values.min()) + 1e-10
    return entropy_scipy(values_stable, uniform)

def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
    x: A num_examples x num_features matrix of features.

    Returns:
    A num_examples x num_examples Gram matrix of examples.
    """
    return x.dot(x.T)

def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

    Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

    Returns:
    A num_examples x num_examples Gram matrix of examples.
    """
    dot_products = x.dot(x.T)
    sq_norms = np.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))

def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

    This is equvialent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.

    Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

    Returns:
    A symmetric matrix with centered columns and rows.
    """
    if not np.allclose(gram, gram.T):
        raise ValueError('Input must be a symmetric matrix.')
    gram = gram.copy()

    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
        # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram


def cka(gram_x, gram_y, debiased=False):
    """Compute CKA.

    Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
    The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


def _get_pwcca_baseline(acts1, acts2, num_datapoints):

    a1shapes = [None, None]
    a2shapes = [None, None]

    if len(acts1.shape) > 2:
        a1shapes = [acts1.shape[0], acts1.shape[1]]
    else:
        a1shapes = acts1.shape 
    
    if len(acts2.shape) > 2:
        a2shapes = [acts2.shape[0], acts2.shape[1]]
    else:
        a2shapes = acts2.shape

    num_datapoints = min(a1shapes[0], a2shapes[0], num_datapoints)
    top_dims = min(a1shapes[1], a2shapes[1])
    b1 = np.random.randn(top_dims, num_datapoints)
    b2 = np.random.randn(top_dims, num_datapoints)

    # Perform SVD
    U1, s1, V1 = np.linalg.svd(b1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(b2, full_matrices=False)

    svacts1 = np.dot(U1.T[:top_dims], b1)
    # can also compute as svacts1 = np.dot(s1[:top_dims]*np.eye(top_dims), V1[:top_dims])
    svacts2 = np.dot(U2.T[:top_dims], b2)
    # can also compute as svacts2 = np.dot(s2[:top_dims]*np.eye(top_dims), V2[:top_dims])

    pwcca_mean, w, _ = pwcca.compute_pwcca(b1, b2, epsilon=1e-10)

    return pwcca_mean

def _get_RV_baseline(acts1, acts2, num_datapoints):

    a1shapes = [None, None]
    a2shapes = [None, None]

    if len(acts1.shape) > 2:
        a1shapes = [acts1.shape[0], acts1.shape[1]]
    else:
        a1shapes = acts1.shape 
    
    if len(acts2.shape) > 2:
        a2shapes = [acts2.shape[0], acts2.shape[1]]
    else:
        a2shapes = acts2.shape

    num_datapoints = min(a1shapes[0], a2shapes[0], num_datapoints)
    top_dims = min(a1shapes[1], a2shapes[1])
    b1 = np.random.randn(num_datapoints, top_dims)
    b2 = np.random.randn(num_datapoints, top_dims)

    aa = gram_linear(b1)
    aa -= np.diag(np.diag(aa))

    bb =  gram_linear(b2) 
    bb -= np.diag(np.diag(bb))

    nom = np.sum(np.multiply(aa, bb))
    denum = np.sqrt( np.sum(np.multiply(aa, aa)) * np.sum(np.multiply(bb, bb)) )
    
    RV_coeff = nom / denum
    return RV_coeff

def _get_abstraction_baseline(acts1, num_datapoints):

    a1shapes = [None, None]

    if len(acts1.shape) > 2:
        a1shapes = [acts1.shape[0], acts1.shape[3]]
    else:
        a1shapes = acts1.shape 
    
    num_datapoints = min(a1shapes[0], num_datapoints)
    
    acts = np.random.randn(a1shapes[1], num_datapoints)

    ent_sum = 0.0
    for i in range(acts.shape[0]):
        ent_sum = (ent_sum*i + entropy_along_data_axis(acts[i]))/(i+1)

    return ent_sum/i
 

def _get_linear_cka_baseline(acts1, acts2, num_datapoints):

    a1shapes = [None, None]
    a2shapes = [None, None]

    if len(acts1.shape) > 2:
        a1shapes = [acts1.shape[0], acts1.shape[3]]
    else:
        a1shapes = acts1.shape 
    
    if len(acts2.shape) > 2:
        a2shapes = [acts2.shape[0], acts2.shape[3]]
    else:
        a2shapes = acts2.shape

    num_datapoints = min(a1shapes[0], a2shapes[0], num_datapoints)
    top_dims = min(a1shapes[1], a2shapes[1])
    b1 = np.random.randn(num_datapoints, top_dims)
    b2 = np.random.randn(num_datapoints, top_dims)

    return cka(gram_linear(b1), gram_linear(b2))

def _get_mean_svcca_baseline(acts1, acts2, num_datapoints, top_dims=20):
    
    a1shapes = [None, None]
    a2shapes = [None, None]

    if len(acts1.shape) > 2:
        a1shapes = [acts1.shape[0], acts1.shape[3]]
    else:
        a1shapes = acts1.shape 
    
    if len(acts2.shape) > 2:
        a2shapes = [acts2.shape[0], acts2.shape[3]]
    else:
        a2shapes = acts2.shape

    num_datapoints = min(a1shapes[0], a2shapes[0], num_datapoints)
    b1 = np.random.randn(a1shapes[1], num_datapoints)
    b2 = np.random.randn(a2shapes[1], num_datapoints)

    # Mean subtract baseline activations
    cb1 = b1 - np.mean(b1, axis=1, keepdims=True)
    cb2 = b2 - np.mean(b2, axis=1, keepdims=True)

    # Perform SVD
    Ub1, sb1, Vb1 = np.linalg.svd(cb1, full_matrices=False)
    Ub2, sb2, Vb2 = np.linalg.svd(cb2, full_matrices=False)

    top_dims = min(sb1.shape[0], sb2.shape[0], top_dims)
    svb1 = np.dot(sb1[:top_dims]*np.eye(top_dims), Vb1[:top_dims])
    svb2 = np.dot(sb2[:top_dims]*np.eye(top_dims), Vb2[:top_dims])

    svcca_baseline = cca_core.get_cca_similarity(svb1, svb2, epsilon=1e-10, verbose=False)
    return np.mean(svcca_baseline["cca_coef1"])

def _get_pwcca_avg_pool(acts1, acts2, num_datapoints):
    if len(acts1.shape) > 2:
        # Activation is convolutianal. Perform average pooling.
        acts1 = np.mean(acts1, axis=(2,3))

    if len(acts2.shape) > 2:
        # Activation is convolutional. Perform average pooling.
        acts2 = np.mean(acts2, axis=(2,3))

    num_datapoints = min(acts1.shape[0], acts2.shape[0], num_datapoints)
    acts1 = acts1.T[:, :num_datapoints]
    acts2 = acts2.T[:, :num_datapoints]

    cacts1 = acts1 - np.mean(acts1, axis=1, keepdims=True)
    cacts2 = acts2 - np.mean(acts2, axis=1, keepdims=True)    
    
    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    top_dims = min(s1.shape[0], s2.shape[0])

    svacts1 = np.dot(U1.T[:top_dims], cacts1)
    # can also compute as svacts1 = np.dot(s1[:top_dims]*np.eye(top_dims), V1[:top_dims])
    svacts2 = np.dot(U2.T[:top_dims], cacts2)
    # can also compute as svacts2 = np.dot(s2[:top_dims]*np.eye(top_dims), V2[:top_dims])

    pwcca_mean, w, _ = pwcca.compute_pwcca(svacts1, svacts2, epsilon=1e-10)

    return pwcca_mean


def _get_mean_svcca_avg_pool(acts1, acts2, num_datapoints, top_dims=20):

    if len(acts1.shape) > 2:
        # Activation is convolutianal. Perform average pooling.
        acts1 = np.mean(acts1, axis=(2,3))

    if len(acts2.shape) > 2:
        # Activation is convolutional. Perform average pooling.
        acts2 = np.mean(acts2, axis=(2,3))

    # Get the maximum number of datapoints that is common in the activation pair
    num_datapoints = min(acts1.shape[0], acts2.shape[0], num_datapoints)
    acts1 = acts1.T[:, :num_datapoints]
    acts2 = acts2.T[:, :num_datapoints]

    cacts1 = acts1 - np.mean(acts1, axis=1, keepdims=True)
    cacts2 = acts2 - np.mean(acts2, axis=1, keepdims=True)
    
    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    top_dims = min(s1.shape[0], s2.shape[0], top_dims)

    svacts1 = np.dot(s1[:top_dims]*np.eye(top_dims), V1[:top_dims])
    # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
    svacts2 = np.dot(s2[:top_dims]*np.eye(top_dims), V2[:top_dims])
    # can also compute as svacts2 = np.dot(U2.T[:20], cacts2)

    svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)
    return np.mean(svcca_results["cca_coef1"])

def _get_RV_avg_pool(acts1, acts2, num_datapoints):
    if len(acts1.shape) > 2:
        # Activation is convolutianal. Perform average pooling.
        acts1 = np.mean(acts1, axis=(2,3))

    if len(acts2.shape) > 2:
        # Activation is convolutional. Perform average pooling.
        acts2 = np.mean(acts2, axis=(2,3))

    # Get the maximum number of datapoints that is common in the activation pair.
    num_datapoints = min(acts1.shape[0], acts2.shape[0], num_datapoints)
    acts1 = acts1[:num_datapoints, :]
    acts2 = acts2[:num_datapoints, :]

    #acts1 = acts1 - np.mean(acts1, axis=1, keepdims=True)
    #acts2 = acts2 - np.mean(acts2, axis=1, keepdims=True)

    aa = gram_linear(acts1)
    aa -= np.diag(np.diag(aa))

    bb = gram_linear(acts2)
    bb -= np.diag(np.diag(bb))

    nom = np.sum(np.multiply(aa, bb))
    denum = np.sqrt( np.sum(np.multiply(aa, aa)) * np.sum(np.multiply(bb, bb)) )
    
    RV_coeff = nom / denum
    return RV_coeff

def _get_abstraction_avg_pool(acts, num_datapoints):

    print(acts.shape)
    if len(acts.shape) > 2:
        # Activation is convolutianal. Perform average pooling.
        acts = np.mean(acts, axis=(2,3))

    # Get the maximum number of datapoints that is common in the activation pair.
    num_datapoints = min(acts.shape[0], num_datapoints)
    acts = acts[:num_datapoints, :]
    
    verbose = False
    kl_along_data = 0.0
    kl_along_neuron = 0.0
    dimensionality = 0.0

    for i in range(acts.shape[1]):
        kl_along_data = (kl_along_data*i + KL_along_data_axis(acts[:, i],verbose=verbose))/(i+1)
   
    for i in range(acts.shape[0]):
        kl_along_neuron = (kl_along_neuron*i + KL_along_neuron_axis(acts[i],verbose=verbose))/(i+1)

    U1, s1, V1 = np.linalg.svd(acts, full_matrices=False)
    s1 += 1e-8
    dimensionality = entropy_scipy(s1)

    abst_dict = {"kl-data":kl_along_data, "kl-neuron":kl_along_neuron, "entropy-sv": dimensionality}

    return abst_dict

def _get_linear_cka_avg_pool(acts1, acts2, num_datapoints):
    if len(acts1.shape) > 2:
        # Activation is convolutianal. Perform average pooling.
        acts1 = np.mean(acts1, axis=(2,3))

    if len(acts2.shape) > 2:
        # Activation is convolutional. Perform average pooling.
        acts2 = np.mean(acts2, axis=(2,3))

    # Get the maximum number of datapoints that is common in the activation pair.
    #print(num_datapoints)
    num_datapoints = min(acts1.shape[0], acts2.shape[0], num_datapoints)
    #print("Num of samples for RV analysis:",num_datapoints)
    acts1 = acts1[:num_datapoints, :]
    acts2 = acts2[:num_datapoints, :]

    return cka(gram_linear(acts1), gram_linear(acts2))

def parse_arch(arch):

    arch_and_optim = arch[:]

    arch_splt = arch.split("_")
    if arch_splt[-1].lower() in ["sgd", "adam"]:
        arch = "_".join(arch_splt[:-1])

    return arch, arch_and_optim

def _remove_activations(arch):
    arch, arch_and_optim = parse_arch(arch)

    act_path = "activations/{0}/".format(arch_and_optim)
    if os.path.exists(act_path):
        clean_command = "rm activations/{0}/*".format(arch_and_optim)
        os.system(clean_command)
    
    
def _produce_activations(arch, checkpoint_epoch):

    arch, arch_and_optim = parse_arch(arch)
    print(arch, arch_and_optim)

    act_path = "activations/{0}/".format(arch_and_optim)
    if not os.path.exists(act_path):
        os.makedirs(act_path)

    clean_command = "rm activations/{0}/*".format(arch_and_optim)
    os.system(clean_command)
    
    command = "python3 main.py -a {0} --resume checkpoints/{2}/checkpoint_{1}.tar \
    --evaluate --collect-acts --activation-dir activations/{2}/".format(arch, checkpoint_epoch, arch_and_optim)
    os.system(command)

def _get_architecture_activations(arch, num_datapoints=500):
    act_dict = defaultdict()
    all_dict = defaultdict(list)

    arch, arch_and_optim = parse_arch(arch)

    for r, d, f in os.walk(os.path.join("activations", arch_and_optim)):
        for act in sorted(f):
            act_path = os.path.join(r, act)
            key = act.split("_")[0]

            if "downsample" in key.lower():
                continue

            all_dict[key].append(act_path)

    model = model = models.__dict__[arch](num_classes=10)
    
    layer_dict_keys = []
    for item, module in model.named_modules():
        if not is_supported(module) or "downsample" in item.lower():
            continue

        layer_dict_keys.append(item)
        
        '''
        splt = item.split(".")
        if splt[-1] == "weight":
            l = ".".join(splt[:-1])
            if l in all_dict:
                dict_keys.append(l)
        '''

    for layer in layer_dict_keys:

        if layer not in all_dict:
            continue

        act_list = []
        num_points = 0

        for it in all_dict[layer]:
            
            if num_points > num_datapoints:
                break
        
            act_list.append(np.load(it))
            num_points += act_list[-1].shape[0]

        act_dict[layer] = np.vstack(act_list)

    return act_dict, layer_dict_keys

def _calculate_sim_from_activations(method, arch, num_datapoints, reference_layer, epoch, baseline=False):
    """ calculates similarity of the activations of all layers to a reference layer for a specific checkpoint """

    act_dict, dict_keys = _get_architecture_activations(arch, num_datapoints)

    if len(dict_keys) <= reference_layer:
        raise ValueError("reference layer index is out of bound")
        return

    all_cca = []
    baseline_cca = []
    all_x = []
    try:
        for l in dict_keys:

            all_x.append(l) 

            current_layer = act_dict[l]
            reference = act_dict[dict_keys[reference_layer]]
            
            corr, baseline_corr = 0,0
            if method == "pwcca":
                corr = _get_pwcca_avg_pool(current_layer, reference, num_datapoints)
                if baseline:
                    baseline_corr =  _get_pwcca_baseline(current_layer, reference, num_datapoints)
            elif method == "svcca":
                corr = _get_mean_svcca_avg_pool(current_layer, reference, num_datapoints)
                if baseline:
                    baseline_corr =  _get_mean_svcca_baseline(current_layer, reference, num_datapoints)
            elif method == "RV":
                corr = _get_RV_avg_pool(current_layer, reference, num_datapoints)
                if baseline:
                    baseline_corr =  _get_RV_baseline(current_layer, reference, num_datapoints)
            else:
                raise ValueError("Invalid method. Options are: pwcca, svcca, RV")
            
            all_cca.append(corr)
            baseline_cca.append(baseline_corr)
            
            print("Comparing:",l,  dict_keys[reference_layer])
            print("  Original avg. similarity: {:.2f}".format( corr ))
            if baseline:
                print("  Baseline avg. similarity: {:.2f}".format(baseline_corr))
            print(" ")

        fig_save_path = "figures/layer_similarity/{3}_{1}_refLayer{2}_{0}_epoch{4}.png".format(num_datapoints, 
                                                                            arch, reference_layer, method, epoch)
        _plot_helper_per_epoch(all_x, [all_cca,], "Similarity", baseline_cca, None, fig_save_path)
        return all_x, all_cca, baseline_cca
    except Exception as e:
        traceback.print_exc()
        return

'''
def _calculate_sv_from_activations(arch, num_datapoints, epoch):
    """ calculates abstractness of the activations of all layers for a specific checkpoint """

    act_dict, dict_keys = _get_architecture_activations(arch)

    kl_data_abs = []
    kl_neuron_abs = []
    entropy_sv_abs = []

    all_x = []
    try:
        for l in dict_keys:
            all_x.append(l) 
            
            current_layer = act_dict[l]            
            abst = 0
            
            print("Calculating:", l)
            abst = _get_abstraction_avg_pool(current_layer, num_datapoints)
            kl_data_abs.append(abst['kl-data'])
            kl_neuron_abs.append(abst['kl-neuron'])
            entropy_sv_abs.append(abst['entropy-sv'])

            if epoch == "baseline":
                print("  Baseline avg. abstractness (kl-data, kl-neuron, entropy-sv):", ["{:.2f}".format(item) for item in list(abst.values())] )
            else:
                print("  Original avg. abstractness (kl-data, kl-neuron, entropy-sv):", ["{:.2f}".format(item) for item in list(abst.values())] ) 
            print(" ")

        fig_save_path_kl_data = "figures/layer_abstractness/{3}/{1}_{0}_{3}_epoch_{2}.png".format(num_datapoints, 
                                                                            arch, epoch, 'kl-data')
        fig_save_path_kl_neuron = "figures/layer_abstractness/{3}/{1}_{0}_{3}_epoch_{2}.png".format(num_datapoints, 
                                                                            arch, epoch, 'kl-neuron')
        fig_save_path_entropy_sv = "figures/layer_abstractness/{3}/{1}_{0}_{3}_epoch_{2}.png".format(num_datapoints, 
                                                                            arch, epoch, 'entropy-sv')

        _plot_helper_per_epoch(all_x, [kl_data_abs,], "Abstractness kl-data", None, None, fig_save_path_kl_data)
        _plot_helper_per_epoch(all_x, [kl_neuron_abs,], "Abstractness kl-neuron", None, None, fig_save_path_kl_neuron)
        _plot_helper_per_epoch(all_x, [entropy_sv_abs,], "Abstractness entropy-sv", None, None, fig_save_path_entropy_sv)

        all_abs = {'kl-data':kl_data_abs, 'kl-neuron':kl_neuron_abs, 'entropy-sv':entropy_sv_abs}
        return all_x, all_abs
    except Exception as e:
        traceback.print_exc()
        return
'''


def _calculate_abs_from_activations(arch, num_datapoints, epoch):
    """ calculates abstractness of the activations of all layers for a specific checkpoint """

    act_dict, dict_keys = _get_architecture_activations(arch)

    kl_data_abs = []
    kl_neuron_abs = []
    entropy_sv_abs = []
    baseline_abs = []
    all_x = []
    try:
        for l in dict_keys:
            all_x.append(l) 
            
            current_layer = act_dict[l]            
            abst = 0
            
            print("Calculating:", l)
            abst = _get_abstraction_avg_pool(current_layer, num_datapoints)
            kl_data_abs.append(abst['kl-data'])
            kl_neuron_abs.append(abst['kl-neuron'])
            entropy_sv_abs.append(abst['entropy-sv'])

            if epoch == "baseline":
                print("  Baseline avg. abstractness (kl-data, kl-neuron, entropy-sv):", ["{:.2f}".format(item) for item in list(abst.values())] )
            else:
                print("  Original avg. abstractness (kl-data, kl-neuron, entropy-sv):", ["{:.2f}".format(item) for item in list(abst.values())] ) 
            print(" ")

        fig_save_path_kl_data = "figures/layer_abstractness/{3}/{1}_{0}_{3}_epoch_{2}.png".format(num_datapoints, 
                                                                            arch, epoch, 'kl-data')
        fig_save_path_kl_neuron = "figures/layer_abstractness/{3}/{1}_{0}_{3}_epoch_{2}.png".format(num_datapoints, 
                                                                            arch, epoch, 'kl-neuron')
        fig_save_path_entropy_sv = "figures/layer_abstractness/{3}/{1}_{0}_{3}_epoch_{2}.png".format(num_datapoints, 
                                                                            arch, epoch, 'entropy-sv')

        _plot_helper_per_epoch(all_x, [kl_data_abs,], "Abstractness kl-data", None, None, fig_save_path_kl_data)
        _plot_helper_per_epoch(all_x, [kl_neuron_abs,], "Abstractness kl-neuron", None, None, fig_save_path_kl_neuron)
        _plot_helper_per_epoch(all_x, [entropy_sv_abs,], "Abstractness entropy-sv", None, None, fig_save_path_entropy_sv)

        all_abs = {'kl-data':kl_data_abs, 'kl-neuron':kl_neuron_abs, 'entropy-sv':entropy_sv_abs}
        return all_x, all_abs
    except Exception as e:
        traceback.print_exc()
        return

def calculate_sim_per_checkpoint(method, arch, num_datapoints, reference_layer, min_epoch, max_epoch, step):
    """ calculates similarity of the activations in a network to a reference layer 
        for each checkpoint and plots all in one figure
    """

    if min_epoch > max_epoch and max_epoch != -1:
        raise ValueError("Epoch interval is invalid: (" + str(min_epoch) + ", " + str(max_epoch))
    if num_datapoints == 500 and method in [ "pwcca", "svcca"]:
        num_datapoints = 3000

    final_dict = locals()
    epochs = []
    used_epochs = []
    used_checkpoints = []
    layer_axis = None
    similarities = []
    baseline = None


    arch, arch_and_optim = parse_arch(arch)

    for r, d, f in os.walk(os.path.join("checkpoints", arch_and_optim)):
        epochs = []
        checkpoints = []
        baseline_file = None
        for file in f:
            if "baseline" in file:
                continue
            else:
                epochs.append(int(file.split(".")[0].split("_")[-1]))
                checkpoints.append(file)

        epochs = np.array(epochs)
        sort_index  = np.argsort(epochs)
        
        checkpoints = np.array(checkpoints)
        epochs = epochs[sort_index]
        checkpoints = checkpoints[sort_index]
        
        min_index = min_epoch<=epochs
        epochs = epochs[min_index]
        checkpoints = checkpoints[min_index]
        if max_epoch != -1:
            max_index = epochs<=max_epoch
            epochs = epochs[max_index]
            checkpoints = checkpoints[max_index]

        print(epochs)
        prev = - step
        for idx, checkpoint_epoch in enumerate(epochs.tolist()):
            
            if prev + step > checkpoint_epoch and checkpoint_epoch != epochs[-1]:
                continue

            prev = checkpoint_epoch

            used_epochs.append(checkpoint_epoch)
            used_checkpoints.append(checkpoints[idx])

            _produce_activations(arch_and_optim, checkpoint_epoch)

            layer_axis, simil, baseline = _calculate_sim_from_activations(method, arch_and_optim, num_datapoints, 
                                reference_layer, checkpoint_epoch, ( idx == (epochs.shape[0]-1) ) )

            _remove_activations(arch_and_optim)

            similarities.append(simil)

    fig_save_path = "figures/layer_similarity/{3}_{1}_refLayer{2}_{0}.svg".format(num_datapoints, 
                                                                                arch_and_optim, reference_layer, method)

    data_save_path = "data/layer_similarity/{3}_{1}_refLayer{2}_{0}.pickle".format(num_datapoints, 
                                                                                arch_and_optim, reference_layer, method)
    _plot_helper_per_epoch(layer_axis, similarities, "Similarity", baseline, used_epochs, fig_save_path)

    final_dict["used_checkpoints"] = used_checkpoints
    final_dict["similarities"] = similarities
    final_dict["baseline"] = baseline
    save_dictionary_to_file(data_save_path, final_dict)

def calculate_sim_between_nets(arch1, arch2, method, epoch1, epoch2, num_datapoints):
    if num_datapoints == 500 and method in [ "pwcca", "svcca"]:
        num_datapoints = 3000

    final_dict = locals()
    
    if epoch1 == -1:
        for r, d, f in os.walk(os.path.join("checkpoints", arch1)):
            epoch1 = sorted([int(file.split(".")[0].split("_")[-1]) for file in f])[-1]

    if epoch2 == -1:
        for r, d, f in os.walk(os.path.join("checkpoints", arch2)):
            epoch2 = sorted([int(file.split(".")[0].split("_")[-1]) for file in f])[-1]    

    _produce_activations(arch1, epoch1)
    acts_dict1, dict_keys1 = _get_architecture_activations(arch1)

    _produce_activations(arch2, epoch2)
    acts_dict2, dict_keys2 = _get_architecture_activations(arch2)

    if len(acts_dict1) != len(acts_dict2):
        raise ValueError("Number of layers does not match for comparison: {0} vs. {1}".
            format(arch1, arch2))

    all_sim = []
    baseline_sim = []
    x_axis = []

    for idx, l in enumerate(dict_keys1):
        x_axis.append(idx+1) #TODO: A better naming for the layer plots.
        
        current_layer = acts_dict1[l]
        reference = acts_dict2[dict_keys2[idx]]

        corr, baseline_corr = 0,0
        if method == "pwcca":
            corr = _get_pwcca_avg_pool(current_layer, reference, num_datapoints)
            baseline_corr =  _get_pwcca_baseline(current_layer, reference, num_datapoints)
        elif method == "svcca":
            corr = _get_mean_svcca_avg_pool(current_layer, reference, num_datapoints)
            baseline_corr =  _get_mean_svcca_baseline(current_layer, reference, num_datapoints)
        elif method == "RV":
            corr = _get_RV_avg_pool(current_layer, reference, num_datapoints)
            baseline_corr =  _get_RV_baseline(current_layer, reference, num_datapoints)
        else:
            raise ValueError("Invalid method. Options are: pwcca, svcca, RV")
        
        all_sim.append(corr)
        baseline_sim.append(baseline_corr)
        
        print("Comparing:",l,  dict_keys2[idx])
        print("  Original avg. similarity: {:.2f}".format( corr ))
        print("  Baseline avg. similarity: {:.2f}".format(baseline_corr))
        print(" ")


    timestamp = time.strftime("%Y%m%d-%H%M%S")
    fig_save_path = "figures/network_similarity/{0}_{1}_vs_{2}_{3}_{4}_{5}.png".format(
                                                    arch1, epoch1, arch2, epoch2, method, timestamp)

    data_save_path = "data/network_similarity/{0}_{1}_vs_{2}_{3}_{4}_{5}.pickle".format(
                                    arch1, epoch1, arch2, epoch2, method, timestamp)
    _plot_helper_per_epoch(x_axis, [all_sim], baseline_sim, None, fig_save_path, False)

    final_dict["similarities"] = all_sim
    final_dict["baseline"] = baseline_sim
    save_dictionary_to_file(data_save_path, final_dict)


'''
def calculate_singular_values_per_layer_per_epoch(arch, num_datapoints, min_epoch, max_epoch, step):
    
    final_dict = locals()
    epochs = []
    used_epochs = []
    used_checkpoints = []
    layer_axis = None
    singular_values = defaultdict(list)
    baseline = None

    arch, arch_and_optim = parse_arch(arch)

    for r, d, f in os.walk(os.path.join("checkpoints", arch_and_optim)):
        epochs = []
        checkpoints = []
        baseline_file = None
        for file in f:
            if "baseline" in file:
                baseline_file = file
                continue
            else:
                epochs.append(int(file.split(".")[0].split("_")[-1]))
                checkpoints.append(file)

        epochs = np.array(epochs)
        sort_index  = np.argsort(epochs)
        
        checkpoints = np.array(checkpoints)
        epochs = epochs[sort_index]
        checkpoints = checkpoints[sort_index]
        
        min_index = min_epoch<=epochs
        epochs = epochs[min_index]
        checkpoints = checkpoints[min_index]
        if max_epoch != -1:
            max_index = epochs<=max_epoch
            epochs = epochs[max_index]
            checkpoints = checkpoints[max_index]

        epochs = epochs.tolist()
        epochs.append('baseline')
        checkpoints = checkpoints.tolist()
        checkpoints.append(baseline_file)

        print(epochs)
        prev = - step
        for idx, checkpoint_epoch in enumerate(epochs):
            
            if checkpoint_epoch != 'baseline':
                if prev + step > checkpoint_epoch and checkpoint_epoch != epochs[-1]:
                    continue

                prev = checkpoint_epoch

            used_epochs.append(checkpoint_epoch)
            used_checkpoints.append(checkpoints[idx])

            _produce_activations(arch_and_optim, checkpoint_epoch)

            dim_axis, sv = _calculate_sv_from_activations(arch_and_optim, num_datapoints, 
                                checkpoint_epoch) 
            
            for key, val in abstractness.items():
                abstractness_values[key].append(val)

            _remove_activations(arch_and_optim)

    for key, val in abstractness_values.items():

        fig_save_path = "figures/layer_abstractness/{2}/{1}_{0}_{2}.svg".format(num_datapoints, arch_and_optim, key)
        #data_save_path = "data/layer_abstractness/{1}_{0}_{2}.pickle".format(num_datapoints, arch_and_optim)

        _plot_helper_per_epoch(layer_axis, val, "Abstractness", None, used_epochs, fig_save_path)
   
'''

def calculate_network_abstraction_per_epoch(arch, num_datapoints, min_epoch, max_epoch, step):

    #final_dict = locals()
    epochs = []
    used_epochs = []
    used_checkpoints = []
    layer_axis = None
    abstractness_values = defaultdict(list)
    baseline = None


    arch, arch_and_optim = parse_arch(arch)

    for r, d, f in os.walk(os.path.join("checkpoints", arch_and_optim)):
        epochs = []
        checkpoints = []
        baseline_file = None
        for file in f:
            if "baseline" in file:
                baseline_file = file
                continue
            else:
                epochs.append(int(file.split(".")[0].split("_")[-1]))
                checkpoints.append(file)

        epochs = np.array(epochs)
        sort_index  = np.argsort(epochs)
        
        checkpoints = np.array(checkpoints)
        epochs = epochs[sort_index]
        checkpoints = checkpoints[sort_index]
        
        min_index = min_epoch<=epochs
        epochs = epochs[min_index]
        checkpoints = checkpoints[min_index]
        if max_epoch != -1:
            max_index = epochs<=max_epoch
            epochs = epochs[max_index]
            checkpoints = checkpoints[max_index]

        epochs = epochs.tolist()
        epochs.append('baseline')
        checkpoints = checkpoints.tolist()
        checkpoints.append(baseline_file)

        print(epochs)
        prev = - step
        for idx, checkpoint_epoch in enumerate(epochs):
            
            if checkpoint_epoch != 'baseline':
                if prev + step > checkpoint_epoch and checkpoint_epoch != epochs[-1]:
                    continue

                prev = checkpoint_epoch

            used_epochs.append(checkpoint_epoch)
            used_checkpoints.append(checkpoints[idx])

            _produce_activations(arch_and_optim, checkpoint_epoch)

            layer_axis, abstractness = _calculate_abs_from_activations(arch_and_optim, num_datapoints, 
                                checkpoint_epoch) 
            
            for key, val in abstractness.items():
                abstractness_values[key].append(val)

            _remove_activations(arch_and_optim)

    for key, val in abstractness_values.items():

        fig_save_path = "figures/layer_abstractness/{2}/{1}_{0}_{2}.svg".format(num_datapoints, arch_and_optim, key)
        #data_save_path = "data/layer_abstractness/{1}_{0}_{2}.pickle".format(num_datapoints, arch_and_optim)

        _plot_helper_per_epoch(layer_axis, val, "Abstractness", None, used_epochs, fig_save_path)
        '''
        final_dict["used_checkpoints"] = used_checkpoints
        final_dict["abstractness_values"] = val
        final_dict["baseline"] = baseline
        save_dictionary_to_file(data_save_path, final_dict)
        '''


def main():
    global parser
    args = parser.parse_args()

    if args.run_mode.lower() == 'in-net':
        calculate_sim_per_checkpoint(args.method, 
                                    args.architecture, 
                                    args.num_datapoints, 
                                    args.reference_layer_idx,
                                    args.min_epoch,
                                    args.max_epoch,
                                    args.step
                                 )

    elif args.run_mode.lower() == 'between-nets':
        calculate_sim_between_nets(args.architecture,
                                    args.architecture2,
                                    args.method,
                                    args.epoch_a1,
                                    args.epoch_a2,
                                    args.num_datapoints
                                )

    elif args.run_mode.lower() == 'abstraction':
        calculate_network_abstraction_per_epoch(args.architecture, args.num_datapoints, args.min_epoch, args.max_epoch, args.step)


if __name__ == '__main__':
    main()
