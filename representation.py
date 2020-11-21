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
from scipy.stats import skewnorm, gaussian_kde
import entropy_estimators

import torch
import torch.nn.functional as F

sys.path.append("svcca")
import cca_core
import pwcca
import models
from utils import *
from autograd_hacks import is_supported

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='Representational Analysis from Activations')

parser.add_argument('--run-mode', '--rm', default='in-net',
                help='Running mode: {in-net, between-nets, abstraction}')
parser.add_argument('--method', '-m', default='RV',
                help='Correlation method to use: mean SVCCA OR PWCCA (default: RV)')
parser.add_argument('--decomposition', '-d', default='evd', help='Decomposition method: svd or evd',
                    choices=['svd', 'evd'])

parser.add_argument('--architecture', '-a', default='resnet18k_16',
                help='architecture (default: cnn5)')
parser.add_argument('--architecture2', '--a2', default='resnet18k_16',
                help='Second architecture (default: resnet18k_16)')
parser.add_argument('--reference-layer-idx', '-r', default=-1, type=int,
                 help='Similarity is calculated against this layer for all layers (default: -1)')
parser.add_argument('--num-datapoints', '-n', default=1024, type=int,
                 help='Number of datapoints used OR Dimension of the activation space (default: 1024)')
parser.add_argument('--min-epoch', default=0, type=int,
                 help='first epoch of the model for the activations (default: 0)')
parser.add_argument('--max-epoch', '-e', default=-1, type=int,
                 help='last epoch of the model for the activations (default: -1)')
parser.add_argument('--epoch-a1', "--e1", default=-1, type=int,
                 help='epoch of the first model for the activations (default: -1)')
parser.add_argument('--epoch-a2', '--e2', default=-1, type=int,
                 help='epoch of the second model for the activations (default: -1)')
parser.add_argument('--checkpoint', '-c', default=None, type=str,
                 help='epoch of the model for the activations')
parser.add_argument('--mode', default="activation", type=str,
                 help='spectral analysis mode')
parser.add_argument('--step', '-s', default=1, type=int,
                 help='step between epochs (default: 1)')
parser.add_argument('--std', default=0.1, type=float, help='std for additive noise on inputs')

def _plot_helper(arr_x, arr_y, baseline_y,  xlabel, ylabel, save_path="svcca_default.png"):
    plt.plot(arr_x, arr_y, label="Layers")
    plt.plot(arr_x, baseline_y, label="Baseline")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')
    plt.grid()
    plt.savefig(save_path)

def _configure_plot_axes(xlabel, ylabel, ylim=None):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)

    if ylim:
        assert isinstance(ylim, tuple) and len(ylim) == 2, "ylim should be a 2 element (int,int) tuple"
        plt.ylim(ylim)

    plt.grid()
    plt.subplots_adjust(bottom=0.25)

def _plot_helper_per_epoch(arr_x, values, baseline, epochs, arch, ylabel, xlabel="Layer", save_path="plot_default.png"):

    fig = plt.figure(figsize=(8,5))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)

    if "dimensionality" not in ylabel:
        if "abstractness" in ylabel:
            plt.ylim((0,0.25))
        else:
            plt.ylim((0,1))
    else:
        plt.ylim((0, 1000))

    plt.grid()
    plt.subplots_adjust(bottom=0.25)

    ### If epochs is None, we need to plot for only given epoch, which is an intermediate result.
    if isinstance(epochs, int) or isinstance(epochs, str):
        plt.title("{}, epoch {}".format(arch, epochs))

        for idx, val in enumerate(values):

            if xlabel == "Dimension Idx" and arr_x.shape[0] > val.shape[0]:
                original_values = val[:]
                val = np.zeros_like(arr_x)
                val[:original_values.shape[0]] = original_values

            plt.plot(arr_x, val, color='b', marker="o", linestyle="-")

            if "dimensionality" in ylabel:
                for i,j in zip(arr_x, val):
                    plt.annotate(str(j),
                        (i,j),
                        textcoords="offset points", # how to position the text
                        xytext=(0,5), # distance from text to points (x,y)
                        ha='center')

            #Since this is an intermediate result, save the figure into a separate folder.
            splt = save_path.split('/')
            splt.insert(-1, 'progress')
            save_path_new = os.path.join(*splt)

            plt.savefig(save_path_new)
            plt.close(fig)
        return


    ### When epochs are given, we are plotting all epochs together.
    plt.title("{}".format(arch))
    c = np.array([item for item in epochs if item!='baseline'])

    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.inferno)
    cmap.set_array([])

    """
        Baseline is given separately when calculating activation similarity.
        This is because for activation similarity, baseline is the similarity of random
        activation, NOT activations calculated from initial weights.

        Baseline is given within values and epochs when calculating abstractness.
        This is because for abstractness, baseline is abstractness of initial weights when
        given real/natural data.
    """

    # If baseline is given sperataley (this should happen when activation similarity is plotted),
    # we plot it using the values in 'baseline' variable.
    if baseline:
        plt.plot(arr_x, baseline,  color='k', linestyle='--', label='baseline')
    for idx, epoch in enumerate(epochs):

        if  xlabel!="Layer" and arr_x.shape[0] > values[idx].shape[0]:
            original_values = values[idx][:]
            values[idx] = np.zeros_like(arr_x)
            values[idx][:original_values.shape[0]] = original_values

        # when baseline is not givenseparately (this should happeb when calculating abstractness),
        # it is included in the epochs array, where we plot it with a different(dashed) line style.

        if epoch == 'baseline':
            plt.plot(arr_x, values[idx],  color='k', linestyle='--', label='baseline')
        else:
            plt.plot(arr_x, values[idx], c=cmap.to_rgba(epoch))

    cbar = plt.colorbar(cmap, ticks=[c[0], c[-1]])
    cbar.set_label("Epoch")
    plt.savefig(save_path)
    plt.close(fig)

def _plot_helper_per_epoch_scatter(arr_x, values, baseline, epochs, arch, ylabel, xlabel="Layer", save_path="plot_default.png"):

    fig = plt.figure(figsize=(8,5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.grid()
    plt.subplots_adjust(bottom=0.25)

    ### If epochs is None, we need to plot for only given epoch, which is an intermediate result.
    if isinstance(epochs, int) or isinstance(epochs, str):
        plt.title("{} epoch {}".format(arch, epochs))
        for idx, val in enumerate(values):

            if xlabel!="Layer" and arr_x.shape[0] > val.shape[0]:
                original_values = val[:]
                val = np.zeros_like(arr_x)
                val[:original_values.shape[0]] = original_values

            plt.scatter(arr_x, val, color='b', marker="o", linestyle="-")

            #Since this is an intermediate result, save the figure into a separate folder.
            splt = save_path.split('/')
            splt.insert(-1, 'progress')
            save_path_new = os.path.join(*splt)

            plt.savefig(save_path_new)
            plt.close(fig)
        return


    ### When epochs are given, we are plotting all epochs together.

    c = np.array([item for item in epochs if item!='baseline'])
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.inferno)
    cmap.set_array([])

    """
    Baseline is given separately when calculating activation similarity.
    This is because for activation similarity, baseline is the similarity of random
    activation, NOT activations calculated from initial weights.

    Baseline is given within values and epochs when calculating abstractness.
    This is because for abstractness, baseline is abstractness of initial weights when
    given real/natural data.
    """

    # If baseline is given sperataley (this should happen when activation similarity is plotted),
    # we plot it using the values in 'baseline' variable.
    if baseline:
        plt.plot(arr_x, baseline,  color='k', linestyle='--', label='baseline')
    for idx, epoch in enumerate(epochs):

        # when baseline is not givenseparately (this should happeb when calculating abstractness),
        # it is included in the epochs array, where we plot it with a different(dashed) line style.

        if epoch == 'baseline':
            continue
            #plt.scatter(arr_x[idx], values[idx],  color='k', linestyle='--', label='baseline')
        else:
            plt.scatter(arr_x[idx], values[idx], c=[cmap.to_rgba(epoch) for item in arr_x[idx]])

    cbar = plt.colorbar(cmap, ticks=[c[0], c[-1]])
    cbar.set_label("Epoch")
    plt.savefig(save_path)
    plt.close(fig)

def _gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q

def _pool_and_flatten(acts, pool=True, max_size=4):
    # Activation is convolutianal. Perform average pooling.
    if len(acts.shape) > 2 and pool:

        """ If feature map is larger than 4x4, downsample to 4x4 representation. """
        if acts.shape[2] > max_size and acts.shape[3] > max_size:
            acts = F.adaptive_avg_pool2d(torch.from_numpy(acts), max_size).numpy()

    acts = acts.reshape((acts.shape[0], -1))

    return acts

def activation_histogram(acts, decomposition, arch, param_name, epoch, bins, first_k=8):

    _dims = min(acts.shape[1], first_k)

    save_path = "figures/layer_abstractness/{}/activation-histogram/{}/{}/{}".format(decomposition, arch, param_name, epoch)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for indx in range(_dims):
        fig = plt.figure(figsize=(8,5))
        fig_path = os.path.join(save_path, "{0}_{1}_epoch{2}_sv{3}.png".format(arch,
                                                        param_name,
                                                        epoch,
                                                        indx
                                                        )
                                                    )

        plt.title("{} Singular Vector {}".format(param_name, indx))
        plt.xlabel("Activation Values")
        plt.ylabel("Frequency")

        if (bins[indx] == None).any():
            _bins = 'sqrt'
        else:
            _bins = bins[indx]

        plt.hist(acts[:, indx], bins=_bins, density=True)

        plt.savefig(fig_path)
        plt.close(fig)


def set_up_bins(ws, num_of_bins):
    low=0
    epoch_bins=[]

    max_val = ws.max(axis=1)

    for i in range(len(ws)):
        layer_bins=[low]
        unique=np.unique(np.squeeze(ws[i].reshape(1,-1)))
        unique=np.sort(np.setdiff1d(unique,layer_bins))
        if unique.size>0:
            for k in range(num_of_bins):
                ind=int(k*(unique.size/num_of_bins))
                layer_bins.append(unique[ind])
            layer_bins.append(unique[-1])
        else:
            layer_bins = None
        epoch_bins.append(np.asarray(layer_bins))

    print('  binning is done')
    bins=np.asarray(epoch_bins)
    return bins

def KL_uniform(values, normalize=True, verbose=False):
    '''
    get distribution of acts. with histogram, then calculate KL div.
    from a uniform distribution.
    '''

    if verbose:
        print(values)

    hist_original, bins = np.histogram(values, bins='sqrt', density=True)
    num_bins = hist_original.shape[0]

    hist_uniform = np.ones_like(hist_original) / num_bins

    _div = entropy_scipy(hist_original + 1e-10, hist_uniform)

    if normalize:
        K = np.log(num_bins)
        _div /= K

    return _div

def KL_normal(values, normalize, skewness, verbose=False):
    '''
    get distribution of acts. with histogram, then calculate KL div.
    from a (skewed) normal dist. with the same mean and std.
    (and skew.) of the actual activation dist.
    '''
    if verbose:
        print(values)

    hist_original = np.histogram(values, bins='sqrt', density=True)

    _mean = values.mean()
    _std = values.std()

    if skewness:
        _median_idx = np.argmax(hist_original[0])
        _median = (hist_original[1][_median_idx] + hist_original[1][_median_idx + 1] )/ 2
        _skew = (_mean - _median)/_std
        normal_values = skewnorm.rvs(_skew, _mean, _std, values.shape[0])

    else:
        normal_values = np.random.normal(_mean, _std, values.shape[0])

    hist_normal = np.histogram(normal_values, bins=hist_original[1], density=True)
    _div = entropy_scipy(hist_original[0]+1e-10, hist_normal[0]+1e-10)

    if normalize:
        bins = len(hist_original[0])
        K = np.log(bins)
        _div /= K

    return _div

def histogram_entropy(values, bins, normalize=True):
    ''' builds histogram to and compute entropy'''

    if (bins == None).any():
        bins = 'sqrt'

    hist, _ = np.histogram(values, bins=bins, density=True)
    _ent = entropy_stable(hist)

    if normalize:
        K = np.log(len(bins))
        _ent /= K

    return _ent

def entropy_stable(values):
    ''' fixes less than or equal to zero values before calculating entropy'''
    values_stable = (values - values.min()) + 1e-10
    return entropy_scipy(values_stable)

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

def cka(gram_x, gram_y, debiased=True):
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

def _get_pwcca_baseline(acts1, acts2):

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

    num_datapoints = min(a1shapes[0], a2shapes[0])
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

def _get_RV_baseline(acts1, acts2):
    '''
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
    '''

    b1 = np.random.randn(*(acts1.shape))
    b2 = np.random.randn(*(acts2.shape))

    num_datapoints = min(b1.shape[0], b2.shape[0])
    b1 = b1[:num_datapoints]
    b2 = b2[:num_datapoints]

    b1 = _pool_and_flatten(b1)
    b2 = _pool_and_flatten(b2)

    aa = gram_linear(b1)
    aa -= np.diag(np.diag(aa))

    bb =  gram_linear(b2)
    bb -= np.diag(np.diag(bb))

    nom = np.sum(np.multiply(aa, bb))
    denum = np.sqrt( np.sum(np.multiply(aa, aa)) * np.sum(np.multiply(bb, bb)) )

    RV_coeff = nom / denum
    return RV_coeff

def _get_linear_cka_baseline(acts1, acts2):

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

    num_datapoints = min(a1shapes[0], a2shapes[0])
    top_dims = min(a1shapes[1], a2shapes[1])
    b1 = np.random.randn(num_datapoints, top_dims)
    b2 = np.random.randn(num_datapoints, top_dims)

    return cka(gram_linear(b1), gram_linear(b2))

def _get_mean_svcca_baseline(acts1, acts2, top_dims=20):

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

    num_datapoints = min(a1shapes[0], a2shapes[0])
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

def _get_pwcca_avg_pool(acts1, acts2):
    if len(acts1.shape) > 2:
        # Activation is convolutianal. Perform average pooling.
        acts1 = np.mean(acts1, axis=(2,3))

    if len(acts2.shape) > 2:
        # Activation is convolutional. Perform average pooling.
        acts2 = np.mean(acts2, axis=(2,3))

    num_datapoints = min(acts1.shape[0], acts2.shape[0])
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
    num_datapoints = min(acts1.shape[0], acts2.shape[0])
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

def _get_RV_avg_pool(acts1, acts2):
    # Get the maximum number of datapoints that is common in the activation pair.
    num_datapoints = min(acts1.shape[0], acts2.shape[0])
    acts1 = acts1[:num_datapoints, :]
    acts2 = acts2[:num_datapoints, :]

    acts1 = _pool_and_flatten(acts1)
    acts2 = _pool_and_flatten(acts2)

    acts1 = acts1 - np.mean(acts1, axis=0, keepdims=True)
    acts2 = acts2 - np.mean(acts2, axis=0, keepdims=True)

    aa = center_gram(gram_linear(acts1))
    aa -= np.diag(np.diag(aa))

    bb = center_gram(gram_linear(acts2))
    bb -= np.diag(np.diag(bb))

    nom = np.sum(np.multiply(aa, bb))
    denum = np.sqrt( np.sum(np.multiply(aa, aa)) * np.sum(np.multiply(bb, bb)) )

    RV_coeff = nom / denum
    return RV_coeff

def _get_sv_avg_pool(acts):
    print(acts.shape)
    if len(acts.shape) > 2:
        # Activation is convolutianal. Perform average pooling.
        acts = np.mean(acts, axis=(2,3))
    # Get the maximum number of datapoints that is common in the activation pair.
    verbose = False
    U1, s1, V1 = np.linalg.svd(acts, full_matrices=False)
    total = np.sum(s1)
    _sum = 0.0
    for i, item in enumerate(s1.tolist()):
        if _sum/total >= 0.50 and item/s1[0] <= 0.01:
            if i==0:
                return s1[0]
            else:
                return s1[:i]
        _sum += item
    return s1

def _get_linear_cka_avg_pool(acts1, acts2, pool_and_flatten=True):
    num_datapoints = min(acts1.shape[0], acts2.shape[0])
    acts1 = acts1[:num_datapoints, :]
    acts2 = acts2[:num_datapoints, :]

    if pool_and_flatten:
        acts1 = _pool_and_flatten(acts1)
        acts2 = _pool_and_flatten(acts2)

    acts1 = acts1 - np.mean(acts1, axis=0, keepdims=True)
    acts2 = acts2 - np.mean(acts2, axis=0, keepdims=True)

    return cka(gram_linear(acts1), gram_linear(acts2))

def _get_spectral_analysis_avg_pool(decomposition, acts, mode):
    if mode == "weight":
        print("  shape :", acts.shape)
        if len(acts.shape) > 2:
            #acts = torch.transpose(acts, 0, 1).flatten(start_dim=1)
            acts = torch.flatten(acts, start_dim=1)
            #acts = torch.mean(acts, (2,3))

        l2norm = torch.norm(acts).detach().cpu().numpy()
        acts = acts.detach().cpu().numpy()

        U1, s1, _ = np.linalg.svd(acts, full_matrices=False)
        s1_squared = s1**2
        s1_squared_non_zero = s1_squared[s1_squared > 1e-10*s1_squared.shape[0]*s1_squared[0]]
        stable_rank = s1_squared_non_zero.sum() / s1_squared[0]
        spectral_norm = s1_squared[0]
        rank = s1_squared_non_zero.shape[0]
        eigen_sum = s1_squared_non_zero.sum()

        spectral_dict = {
                    "l2_norm": l2norm,
                    "spectral_norm": spectral_norm,
                    "stable_rank" : stable_rank,
                    "eigen_sum": eigen_sum,
                    "rank": rank
            }

        return spectral_dict

    else:
        print("  shape :", acts.shape[1:])
        acts = _pool_and_flatten(acts, pool=(decomposition=="svd"))
        print("  neurons:", acts.shape[1])
        # Center the columns
        acts = acts - np.mean(acts, axis=0, keepdims=True)
        verbose = False

        if decomposition == "svd":
            ''' for singular val. decomp. '''
            U1, s1, _ = np.linalg.svd(acts.astype(np.float64), full_matrices=False)
            s1_squared = s1**2

        elif decomposition == "evd":
            ''' For eigen decoms. '''
            s1_squared, U1 = np.linalg.eigh(center_gram(gram_linear(acts.astype(np.float64)), False))
            #s1_squared, U1 = np.linalg.eigh(gram_linear(acts.astype(np.float64)))
            s1_squared = np.flip(s1_squared)
            U1 = np.flip(U1, axis=1)

            #TODO: some eigenvalues are negative. find out why. this is not a fix.
            s1_squared[s1_squared<0] = 0.0
            s1 = s1_squared**0.5

        _max_dim = 0
        _max_ratio = 0.99
        _total = np.sum(s1_squared)
        _total_sv = np.sum(s1)
        _total = np.sum(s1_squared)
        _sum = 0.0
        dimensionality = len(s1_squared)
        for i, item in enumerate(s1_squared.tolist()):
            if i>0:
                coverage = _sum/_total
                # selection statistic that covers large percentage
                if coverage > 0.99:
                    dimensionality = i
                    break
            _sum += item


        s1_squared_non_zero = s1_squared[:dimensionality] #s1_squared[s1_squared > 1e-10*s1_squared.shape[0]*s1_squared[0]]
        stable_rank = s1_squared_non_zero.sum() / s1_squared[0]
        spectral_norm = s1_squared[0]
        rank = s1_squared_non_zero.shape[0]
        eigen_sum = s1_squared_non_zero.sum()
        spectral_dict = {
                    "spectral_norm": spectral_norm,
                    "stable_rank" : stable_rank,
                    "eigen_sum": eigen_sum,
                    "rank": rank
            }
        return spectral_dict

def _get_abstraction_avg_pool(decomposition, acts, arch, param_name, epoch):
    print("  shape :", acts.shape[1:])
    acts = _pool_and_flatten(acts, pool=(decomposition == "svd"))
    print("  neurons:", acts.shape[1])

    # Center the columns
    acts = acts - np.mean(acts, axis=0, keepdims=True)
    verbose = False
    avg_entropy = 0.0
    #w_avg_entropy = 0.0
    total_entropy = 0.0
    entropy_scatter = []
    avg_KLdiv = 0.0
    #w_avg_KLdiv = 0.0
    total_KLdiv = 0.0
    KLdiv_scatter = []
    avg_abstractness = 0.0
    #w_avg_abstractness = 0.0
    total_abstractness = 0.0
    abstractness_scatter = []
    compressability = 0.0

    if decomposition == "svd":
        ''' for singular val. decomp. '''
        U1, s1, _ = np.linalg.svd(acts.astype(np.float64), full_matrices=False)
        s1_squared = s1**2

    elif decomposition == "evd":
        ''' For eigen decoms. '''
        s1_squared, U1 = np.linalg.eigh(center_gram(gram_linear(acts.astype(np.float64)), False))
        s1_squared = np.flip(s1_squared)
        U1 = np.flip(U1, axis=1)
        #TODO: some eigenvalues are negative. find out why. this is not a fix.
        s1_squared[s1_squared<0] = 0.0
        s1 = s1_squared**0.5

    dimensionality = None
    _total = np.sum(s1_squared)
    _sum = 0.0
    for i, item in enumerate(s1_squared.tolist()):
        if i>0:
            coverage = _sum/_total
            # selection statistic that covers large percentage
            if coverage > 0.99:
                dimensionality = i
                break
        _sum += item
    start_index = 0

    reduced_s1_squared = s1_squared[start_index:dimensionality]
    reduced_s1 = np.diag(s1[start_index:dimensionality])
    reduced_U1 = U1[:,start_index:dimensionality]
    reduced_acts = np.matmul(reduced_U1, reduced_s1)
    reduced_acts -= reduced_acts.min(axis=1, keepdims=True)
    dimensionality -= start_index

    _bins = set_up_bins(reduced_acts.T, int(reduced_acts.shape[0]**0.5))
    '''Takes TOO MUCH TIME, used for visualization. Uncomment only if a different set of activations is used '''
    #activation_histogram(reduced_acts, decomposition, arch, param_name, epoch, _bins)

    """
    log_entropy_sum = 0.0
    log_KLdiv_sum = 0.0
    log_abstractness_sum = 0.0
    """
    for i in range(reduced_acts.shape[1]):

        _ent = histogram_entropy(reduced_acts[:, i], _bins[i], normalize=True)
        _kl_normal = 1 - _ent #KL_uniform(reduced_acts[:, i], normalize=True, verbose=verbose)
        _abst = _kl_normal * _ent

        #log_entropy_sum += np.log1p(_ent)
        #w_avg_entropy += _ent * reduced_weights[i]
        total_entropy += _ent
        entropy_scatter.append(_ent)

        #log_KLdiv_sum += np.log1p(_kl_normal)
        #w_avg_KLdiv += _kl_normal * reduced_weights[i]
        total_KLdiv += _kl_normal
        KLdiv_scatter.append(_kl_normal)

        #log_abstractness_sum += np.log1p(_abst)
        #w_avg_abstractness += _abst * reduced_weights[i]
        total_abstractness += _abst
        abstractness_scatter.append(_abst)

    entropy_scatter = np.array(entropy_scatter)
    KLdiv_scatter = np.array(KLdiv_scatter)
    abstractness_scatter = np.array(abstractness_scatter)

    #avg_log_entropy = log_entropy_sum / dimensionality
    #geo_avg_entropy = np.expm1(avg_log_entropy)
    w_avg_entropy = (entropy_scatter*reduced_s1_squared).sum() / _total
    avg_entropy = total_entropy / dimensionality

    #avg_log_KLdiv = log_KLdiv_sum / dimensionality
    #geo_avg_KLdiv = np.expm1(avg_log_KLdiv)
    w_avg_KLdiv = (KLdiv_scatter*reduced_s1_squared).sum() / _total
    avg_KLdiv =  total_KLdiv / dimensionality #KLdiv_sum/dimensionality

    #avg_log_abstractness = log_abstractness_sum / dimensionality
    #geo_avg_abstractness = np.expm1(avg_log_abstractness)  # avg_entropy * avg_KLdiv
    w_avg_abstractness = (abstractness_scatter *reduced_s1_squared ).sum() / _total
    avg_abstractness = total_abstractness / dimensionality  #abstractness_sum/dimensionality

    abst_dict = {
                "avg-entropy" : avg_entropy,
                "scatter-entropy" : entropy_scatter.tolist(),
                #"geo-avg-entropy" : geo_avg_entropy,
                "weighted-avg-entropy" : w_avg_entropy,
                "avg-KLdiv" : avg_KLdiv,
                "scatter-KLdiv" : KLdiv_scatter.tolist(),
                #"geo-avg-KLdiv" : geo_avg_KLdiv,
                "weighted-avg-KLdiv" : w_avg_KLdiv,
                "avg-abstractness" : avg_abstractness,
                "scatter-abstractness": abstractness_scatter.tolist(),
                #"geo-avg-abstractness": geo_avg_abstractness,
                "weighted-avg-abstractness": w_avg_abstractness,
                "dimensionality": dimensionality,
        }


    return abst_dict

def parse_arch(arch):
    arch_and_optim = arch[:]
    if arch_and_optim in model_names:
        return arch, arch_and_optim
    arch_splt = arch.split("_")
    for i in range(1, len(arch_splt)):
        arch = "_".join(arch_splt[:-i])
        if arch in model_names:
            return arch, arch_and_optim
    return arch, arch_and_optim

def _remove_activations(arch):
    arch, arch_and_optim = parse_arch(arch)
    act_path = "activations/{0}/".format(arch_and_optim)
    if os.path.exists(act_path):
        clean_command = "rm activations/{0}/*".format(arch_and_optim)
        os.system(clean_command)

def _produce_activations(arch, checkpoint_epoch, num_datapoints, noise_std=None):
    arch, arch_and_optim = parse_arch(arch)
    print(arch, arch_and_optim)
    act_path = "activations/{0}/".format(arch_and_optim)

    if not os.path.exists(act_path):
        os.makedirs(act_path)
    clean_command = "rm activations/{0}/*".format(arch_and_optim)
    os.system(clean_command)

    command = "python3 main.py -a {0} --resume checkpoints/{2}/checkpoint_{1}.tar \
        --evaluate --collect-acts --activation-dir activations/{2}/ --activation-size {3}" \
        .format(arch, checkpoint_epoch, arch_and_optim, num_datapoints)
    if noise_std:
        command += " --input-noise --sigma {}".format(noise_std)
    os.system(command)

def _get_architecture_activations(arch):
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
    model = models.__dict__[arch](num_classes=10)
    #model.print_arch()
    #sys.exit()
    layer_dict_keys = [item[0] for item in model.named_modules()]
    """
    for item, module in model.named_modules():
        #if not is_supported_activation(module) or "downsample" in item.lower():
        #    continue

        layer_dict_keys.append(item)
        '''
        splt = item.split(".")
        if splt[-1] == "weight":
            l = ".".join(splt[:-1])
            if l in all_dict:
                dict_keys.append(l)
        '''
    """

    for layer in layer_dict_keys:
        if layer not in all_dict:
            continue
        act_list = []
        num_points = 0
        for it in all_dict[layer]:
            act_list.append(np.load(it))
            num_points += act_list[-1].shape[0]
        act_dict[layer] = np.vstack(act_list)

    return act_dict, list(act_dict.keys())

def _calculate_similarity_from_activations(method, arch, num_datapoints, reference_layer, epoch, baseline=False):
    """ calculates similarity of the activations of all layers to a reference layer for a specific checkpoint """

    act_dict, dict_keys = _get_architecture_activations(arch)

    if len(dict_keys) <= reference_layer:
        raise ValueError("reference layer index is out of bound")
        return

    all_cca = []
    baseline_cca = []
    all_x = []
    try:
        for idx, k in enumerate(dict_keys):
            all_x.append(k)
            current_layer = act_dict[k]
            reference = act_dict[dict_keys[reference_layer]]
            corr, baseline_corr = 0,0
            if method == "pwcca":
                corr = _get_pwcca_avg_pool(current_layer, reference)
                if baseline:
                    baseline_corr =  _get_pwcca_baseline(current_layer, reference)
            elif method == "svcca":
                corr = _get_mean_svcca_avg_pool(current_layer, reference)
                if baseline:
                    baseline_corr =  _get_mean_svcca_baseline(current_layer, reference)
            elif method == "RV":
                corr = _get_linear_cka_avg_pool(current_layer, reference)
                if baseline:
                    baseline_corr =  _get_RV_baseline(current_layer, reference)
            else:
                raise ValueError("Invalid method. Options are: pwcca, svcca, RV")
            all_cca.append(corr)
            baseline_cca.append(baseline_corr)
            print("Comparing:",dict_keys[idx],  dict_keys[reference_layer])
            print("  Original avg. similarity: {:.2f}".format( corr ))
            if baseline:
                print("  Baseline avg. similarity: {:.2f}".format(baseline_corr))
            print(" ")

        fig_save_path = "figures/layer_similarity/{3}_{1}_refLayer{2}_{0}_epoch{4}.png".format(num_datapoints,
                                                                            arch, reference_layer, method, epoch)

        _plot_helper_per_epoch(all_x, [all_cca,], baseline_cca, epoch, arch, "RV Similarity", "Layer", fig_save_path)
        return all_x, all_cca, baseline_cca
    except Exception as e:
        traceback.print_exc()
        return

def _calculate_sv_from_activations(arch, num_datapoints, epoch):
    """ calculates singular values of the activations of all layers for a specific checkpoint """

    act_dict, dict_keys = _get_architecture_activations(arch)

    sv_layers = defaultdict()
    try:
        for l in dict_keys:
            current_layer = act_dict[l]
            sv = 0

            print("Calculating:", l)
            sv = _get_sv_avg_pool(current_layer, num_datapoints)
            sv_layers[l] = sv

            fig_save_path = "figures/singular_values/{1}_{3}_epoch-{2}_{0}.png".format(num_datapoints,
                                                                                arch, epoch, l)
            x_dim = np.arange(len(sv))
            _plot_helper_per_epoch(x_dim, [sv,], None, epoch,  arch, "Singular Value", "Dimension Idx",fig_save_path)

        return sv_layers
    except Exception as e:
        traceback.print_exc()
        return

def _calculate_abstraction_from_activations(decomposition, arch, num_datapoints, epoch):
    """ calculates abstractness of the activations of all layers for a specific checkpoint """

    act_dict, dict_keys = _get_architecture_activations(arch)

    abstraction_dict = defaultdict(list)
    scatter_values = defaultdict(list)

    x_dimension = []
    x_dimension_scatter = defaultdict(list)
    try:
        for idx, layer_name in enumerate(dict_keys):

            x_dimension.append(layer_name)
            current_layer = act_dict[layer_name]

            print("Calculating:", layer_name)
            abst = _get_abstraction_avg_pool(decomposition, current_layer, arch, layer_name, epoch)

            for k,v in abst.items():
                if "scatter" in k:
                    x_dimension_scatter[k] += [dict_keys[idx] for item in v]
                    scatter_values[k] += v
                else:
                    abstraction_dict[k].append(v)

            print("  dims.  :",  "{}".format(int(abst["dimensionality"])))


        for k, scatter_val in scatter_values.items():
            scatter_save_path = "figures/layer_abstractness/{4}/{3}/{1}_{0}_{3}_epoch_{2}_SCATTER.png".format(
                                                                                num_datapoints,
                                                                                arch,
                                                                                epoch, k,
                                                                                decomposition)
            _plot_helper_per_epoch_scatter(x_dimension_scatter[k],
                                        [scatter_val,],
                                         None, epoch, arch,
                                         "{}".format(k),
                                         "Layer",
                                         scatter_save_path)


        for k,v in abstraction_dict.items():
            fig_save_path = "figures/layer_abstractness/{4}/{3}/{1}_{0}_{3}_epoch_{2}.png".format(
                                                                                num_datapoints,
                                                                                arch,
                                                                                epoch, k,
                                                                                decomposition)
            _plot_helper_per_epoch(x_dimension,
                                    [v,],
                                     None,
                                     epoch,  arch,
                                     "{}".format(k),
                                     "Layer",
                                     fig_save_path)

        return x_dimension, abstraction_dict, x_dimension_scatter, scatter_values
    except Exception as e:
        traceback.print_exc()
        return

def calculate_similarity_per_checkpoint(method, arch, num_datapoints, reference_layer, min_epoch, max_epoch, step):
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

            _produce_activations(arch_and_optim, checkpoint_epoch, num_datapoints)

            layer_axis, simil, baseline = _calculate_similarity_from_activations(method, arch_and_optim,
                                num_datapoints, reference_layer, checkpoint_epoch, ( idx == (epochs.shape[0]-1) ) )

            _remove_activations(arch_and_optim)

            similarities.append(simil)

    fig_save_path = "figures/layer_similarity/{3}_{1}_refLayer{2}_{0}.png".format(num_datapoints,
                                                                                arch_and_optim, reference_layer, method)

    data_save_path = "data/layer_similarity/{3}_{1}_refLayer{2}_{0}.pickle".format(num_datapoints,
                                                                                arch_and_optim, reference_layer, method)
    _plot_helper_per_epoch(layer_axis, similarities, baseline, used_epochs, arch, "RV Similarity", "Layer", fig_save_path)

    final_dict["used_checkpoints"] = used_checkpoints
    final_dict["similarities"] = similarities
    final_dict["baseline"] = baseline
    save_dictionary_to_file(data_save_path, final_dict)

"""
def calculate_similarity_between_nets(arch1, arch2, method, epoch1, epoch2, num_datapoints):
    if num_datapoints == 500 and method in [ "pwcca", "svcca"]:
        num_datapoints = 3000

    final_dict = locals()

    if epoch1 == -1:
        for r, d, f in os.walk(os.path.join("checkpoints", arch1)):
            epoch1 = sorted([int(file.split(".")[0].split("_")[-1]) for file in f])[-1]

    if epoch2 == -1:
        for r, d, f in os.walk(os.path.join("checkpoints", arch2)):
            epoch2 = sorted([int(file.split(".")[0].split("_")[-1]) for file in f])[-1]

    _produce_activations(arch1, epoch1, num_datapoints)
    acts_dict1, dict_keys1 = _get_architecture_activations(arch1)

    _produce_activations(arch2, epoch2, num_datapoints)
    acts_dict2, dict_keys2 = _get_architecture_activations(arch2)

    if len(acts_dict1) != len(acts_dict2):
        raise ValueError("Number of layers does not match for comparison: {0} vs. {1}".
            format(arch1, arch2))

    all_sim = []
    baseline_sim = []
    x_axis = []

    for idx, l in enumerate(dict_keys1):
        x_axis.append(idx) #TODO: A better naming for the layer plots.

        current_layer = acts_dict1[l]
        reference = acts_dict2[dict_keys2[idx]]

        corr, baseline_corr = 0,0
        if method == "pwcca":
            corr = _get_pwcca_avg_pool(current_layer, reference)
            baseline_corr =  _get_pwcca_baseline(current_layer, reference)
        elif method == "svcca":
            corr = _get_mean_svcca_avg_pool(current_layer, reference)
            baseline_corr =  _get_mean_svcca_baseline(current_layer, reference)
        elif method == "RV":
            corr = _get_RV_avg_pool(current_layer, reference)
            baseline_corr =  _get_RV_baseline(current_layer, reference)
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
"""

def calculate_singular_values_per_layer_per_epoch(arch, num_datapoints, min_epoch, max_epoch, step):

    epochs = []
    used_epochs = []
    used_checkpoints = []
    singular_values_all_epochs = defaultdict(list)
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

            _produce_activations(arch_and_optim, checkpoint_epoch, num_datapoints)

            sv_dict_current_epoch = _calculate_sv_from_activations(arch_and_optim,
                                num_datapoints, checkpoint_epoch)

            for _layer, _sv_list in sv_dict_current_epoch.items():
                singular_values_all_epochs[_layer].append(_sv_list)

            _remove_activations(arch_and_optim)

    for _layer, _list_sv_list in singular_values_all_epochs.items():
        _max_size = 0
        for item in _list_sv_list:
            if len(item)>_max_size:
                _max_size = len(item)

        fig_save_path = "figures/singular_values/{1}_{2}_sv_{0}.png".format(num_datapoints, arch_and_optim, _layer)
        #data_save_path = "data/layer_abstractness/{1}_{0}_{2}.pickle".format(num_datapoints, arch_and_optim)

        x_axis = np.arange(_max_size)
        _plot_helper_per_epoch(x_axis, _list_sv_list, None, used_epochs, arch, "Singular Value", "Dimension Idx",  fig_save_path)

def calculate_network_abstraction_per_epoch(arch, decomposition, num_datapoints, min_epoch, max_epoch, step):

    #final_dict = locals()
    epochs = []
    used_epochs = []
    used_checkpoints = []
    layer_axis = None
    abstractness_values = defaultdict(list)
    layer_axis_scatter_val = defaultdict(list)
    abstractness_scatter_val = defaultdict(list)
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

        abstractness_epoch_stack = defaultdict(list)
        epoch_stack = []
        used_epochs_stack = []
        for idx, checkpoint_epoch in enumerate(epochs):

            if checkpoint_epoch != 'baseline':
                if prev + step > checkpoint_epoch and checkpoint_epoch != epochs[-1]:
                    continue

                epoch_stack.append(checkpoint_epoch)
                prev = checkpoint_epoch
            else:
                used_epochs_stack.append("baseline")

            used_epochs.append(checkpoint_epoch)
            used_checkpoints.append(checkpoints[idx])

            _produce_activations(arch_and_optim, checkpoint_epoch, num_datapoints)

            layer_axis, abstractness, layer_axis_scatter, abstractness_scatter = _calculate_abstraction_from_activations(
                        decomposition,
                        arch_and_optim,
                        num_datapoints,
                        checkpoint_epoch)

            for key, val in abstractness_scatter.items():
                layer_axis_scatter_val[key].append(layer_axis_scatter[key])
                abstractness_scatter_val[key].append(val)

            for key, val in abstractness.items():
                if checkpoint_epoch == "baseline":
                    abstractness_values[key].append(val)
                else:
                    abstractness_epoch_stack[key].append(val)

                if len(epoch_stack) == 1 :
                    _mn = np.array( abstractness_epoch_stack[key] ).mean(axis=0).tolist()
                    abstractness_values[key].append(_mn)

            if len(epoch_stack) == 1 :
                used_epochs_stack.append(np.array(epoch_stack).mean())
                epoch_stack = []
                abstractness_epoch_stack.clear()

            _remove_activations(arch_and_optim)

    for key, val in abstractness_values.items():

        fig_save_path = "figures/layer_abstractness/{3}/{2}/{1}_{0}_{2}.png".format(num_datapoints,
                                                                                    arch_and_optim,
                                                                                    key,
                                                                                    decomposition)

        _plot_helper_per_epoch(layer_axis,
                                val,
                                None,
                                used_epochs_stack, arch,
                                "{}".format(key),
                                "Layer",
                                fig_save_path)


    for key, val in abstractness_scatter_val.items():
        scatter_save_path = "figures/layer_abstractness/{3}/{2}/{1}_{0}_{2}_SCATTER.png".format(num_datapoints,
                                                                                                arch_and_optim,
                                                                                                key,
                                                                                                decomposition)
        _plot_helper_per_epoch_scatter(layer_axis_scatter_val[key],
                                        val,
                                        None,
                                        used_epochs, arch,
                                        "{}".format(key),
                                        "Layer",
                                        scatter_save_path)

def _get_model_weights(architecture, checkpoint_epoch):
    #arch, arch_and_optim = parse_arch(architecture)
    #model = models.__dict__[arch](num_classes=10)

    checkpoint_path = "checkpoints/{}/checkpoint_{}.tar".format(architecture, checkpoint_epoch)
    checkpoint = torch.load(checkpoint_path)
    #model.load_state_dict(checkpoint['state_dict'])

    keys = []
    weights_dict = defaultdict()
    for name,param in checkpoint['state_dict'].items():
        if "weight" not in name or len(param.shape) < 2:
            continue
        new_name = ".".join(name.split(".")[:-1])
        keys.append(new_name)
        weights_dict[new_name] = param

    return weights_dict, keys

def calculate_spectral_analysis_for_epoch(architecture,
                                            decomposition,
                                            num_datapoints,
                                            checkpoint_epoch,
                                            mode
                                    ):
    arch, arch_and_optim = parse_arch(architecture)
    if mode == "weight":
        dict_vals, dict_keys = _get_model_weights(architecture, checkpoint_epoch)
    else:
        _produce_activations(arch_and_optim, checkpoint_epoch, num_datapoints)
        dict_vals, dict_keys = _get_architecture_activations(arch_and_optim)

    x_dimension = []
    spectral_dict_per_layer = defaultdict(list)
    for idx, layer_name in enumerate(dict_keys):

        x_dimension.append(layer_name)
        current_layer = dict_vals[layer_name]

        print("Calculating:", layer_name)

        spectral_dict_for_layer = _get_spectral_analysis_avg_pool(decomposition, current_layer, mode)
        for k,v in spectral_dict_for_layer.items():
            spectral_dict_per_layer[k].append(v)

    _remove_activations(arch_and_optim)

    for k,v in spectral_dict_per_layer.items():
        plt.title("{}, epoch {}, {}, {}".format(arch_and_optim, checkpoint_epoch, k, mode))
        _configure_plot_axes("Layers", k)
        plt.plot(x_dimension, v)
        plt.savefig("figures/layer_abstractness/{}/spectral-analysis/progress/{}_epoch_{}_{}_{}.png".format(decomposition, arch_and_optim, checkpoint_epoch,k,mode))
        plt.clf()


    return x_dimension, spectral_dict_per_layer

def calculate_spectral_analysis(architecture,
                                decomposition,
                                num_datapoints,
                                min_epoch,
                                max_epoch,
                                step,
                                mode):

    assert mode in ["activation", "weight"], "Analysis mode should be one of these: (activation, weight)"
    epochs = []
    used_epochs = []
    used_checkpoints = []
    layer_axis = None
    spectral_values = defaultdict(list)
    baseline = None

    arch, arch_and_optim = parse_arch(architecture)

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
            layer_axis, spectral_dict_for_epoch = calculate_spectral_analysis_for_epoch(arch_and_optim, decomposition, num_datapoints, checkpoint_epoch, mode)
            for k,v in spectral_dict_for_epoch.items():
                spectral_values[k].append(v)


    for k, spectral_list in spectral_values.items():
        fig = plt.figure(figsize=(8,5))
        plt.title("{}, {}, {}".format(arch_and_optim, k, mode))
        _configure_plot_axes("Layers", k)

        c = np.array([item for item in used_epochs if item!='baseline'])
        norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.inferno)
        cmap.set_array([])

        for idx, val in enumerate(spectral_list):
            epoch = used_epochs[idx]
            if epoch == 'baseline':
                plt.plot(layer_axis, val,  color='k', linestyle='--', label='baseline')
            else:
                plt.plot(layer_axis, val, c=cmap.to_rgba(epoch))

        cbar = plt.colorbar(cmap, ticks=[c[0], c[-1]])
        cbar.set_label("Epoch")
        save_path = "figures/layer_abstractness/{}/spectral-analysis/{}_{}_{}.png".format(decomposition, arch_and_optim, k, mode)
        plt.savefig(save_path)
        plt.clf()

def calculate_noise_robustness_for_epoch(arch, checkpoint_epoch, num_datapoints, noise_std=0.1):

    _produce_activations(arch, checkpoint_epoch, num_datapoints, noise_std=noise_std)
    acts_dict_noise, dict_keys = _get_architecture_activations(arch)

    _produce_activations(arch, checkpoint_epoch, num_datapoints)
    acts_dict, dict_keys = _get_architecture_activations(arch)

    #feature_CKA = []
    feature_MSE = []
    x_axis = []

    for idx, l in enumerate(dict_keys):
        print("Comparing:",l)
        x_axis.append(l)

        feature_map = acts_dict[l]
        features_map_noise = acts_dict_noise[l]

        mse = ((feature_map - features_map_noise)**2)
        normalize = (feature_map**2).mean()
        mse /= normalize

        feature_MSE.append( mse.mean() )
        #feature_CKA.append( _get_linear_cka_avg_pool(feature_map, features_map_noise) )

    _remove_activations(arch)
    MSE_save_path = "figures/layer_abstractness/noise_robustness/progress/{0}_{1}_noise_robustness_sigma{2}_MSE.png".format(arch, checkpoint_epoch, noise_std)
    #CKA_save_path = "figures/layer_abstractness/noise_robustness/{0}_{1}_noise_robustness_sigma{2}_CKA.png".format(arch, checkpoint_epoch, noise_std)

    fig = plt.figure(figsize=(8,5))

    plt.title("{}, epoch {}, noise std. {}".format(arch, checkpoint_epoch, noise_std))
    _configure_plot_axes("Layers", "MSE")
    plt.plot(x_axis, feature_MSE)
    plt.savefig(MSE_save_path)

    plt.clf()
    return x_axis, feature_MSE

    '''
    plt.title("{}, epoch {}, noise std. {}".format(arch, checkpoint_epoch, noise_std))
    _configure_plot_axes("Layers", "CKA similarity", (0,1))
    plt.plot(x_axis, feature_CKA)
    plt.savefig(CKA_save_path)

    plt.close(fig)
    '''

def calculate_noise_robustness(arch, min_epoch, max_epoch, num_datapoints, noise_std, step):
    epochs = []
    used_epochs = []
    used_checkpoints = []
    layer_axis = None
    robustness_values = []
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

        abstractness_epoch_stack = defaultdict(list)
        epoch_stack = []
        used_epochs_stack = []
        for idx, checkpoint_epoch in enumerate(epochs):

            if checkpoint_epoch != 'baseline':
                if prev + step > checkpoint_epoch and checkpoint_epoch != epochs[-1]:
                    continue

                epoch_stack.append(checkpoint_epoch)
                prev = checkpoint_epoch
            else:
                used_epochs_stack.append("baseline")

            used_epochs.append(checkpoint_epoch)
            used_checkpoints.append(checkpoints[idx])
            layer_axis, robustness = calculate_noise_robustness_for_epoch(arch_and_optim, checkpoint_epoch, num_datapoints, noise_std)
            robustness_values.append(robustness)

    plt.title("{}, noise std. {}".format(arch_and_optim, noise_std))
    fig = plt.figure(figsize=(8,5))
    _configure_plot_axes("Layers", "MSE")

    c = np.array([item for item in used_epochs if item!='baseline'])
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.inferno)
    cmap.set_array([])

    for idx, val in enumerate(robustness_values):
        epoch = used_epochs[idx]
        if epoch == 'baseline':
            plt.plot(layer_axis, val,  color='k', linestyle='--', label='baseline')
        else:
            plt.plot(layer_axis, val, c=cmap.to_rgba(epoch))

    cbar = plt.colorbar(cmap, ticks=[c[0], c[-1]])
    cbar.set_label("Epoch")
    MSE_save_path = "figures/layer_abstractness/noise_robustness/{0}_noise_robustness_sigma{1}_MSE.png".format(arch_and_optim, noise_std)
    plt.savefig(MSE_save_path)
    plt.clf()

def main():
    global parser
    args = parser.parse_args()

    if args.run_mode.lower() == 'in-net':
        calculate_similarity_per_checkpoint(args.method,
                                    args.architecture,
                                    args.num_datapoints,
                                    args.reference_layer_idx,
                                    args.min_epoch,
                                    args.max_epoch,
                                    args.step
                                 )

    elif args.run_mode.lower() == 'between-nets':
        calculate_similarity_between_nets(args.architecture,
                                    args.architecture2,
                                    args.method,
                                    args.epoch_a1,
                                    args.epoch_a2,
                                    args.num_datapoints
                                )

    elif args.run_mode.lower() == 'abstraction':
        calculate_network_abstraction_per_epoch(
                                    args.architecture,
                                    args.decomposition,
                                    args.num_datapoints,
                                    args.min_epoch,
                                    args.max_epoch,
                                    args.step)


    elif args.run_mode.lower() == 'singular-value':
        calculate_singular_values_per_layer_per_epoch(args.architecture,
                                    args.num_datapoints,
                                    args.min_epoch,
                                    args.max_epoch,
                                    args.step)

    elif args.run_mode.lower() == 'spectral-analysis':
        calculate_spectral_analysis(args.architecture,
                                args.decomposition,
                                args.num_datapoints,
                                args.min_epoch,
                                args.max_epoch,
                                args.step,
                                args.mode)

    elif args.run_mode.lower() == 'noise-robustness':
        calculate_noise_robustness(args.architecture,
                                    args.min_epoch,
                                    args.max_epoch,
                                    args.num_datapoints,
                                    args.std,
                                    args.step)

if __name__ == '__main__':
    main()
