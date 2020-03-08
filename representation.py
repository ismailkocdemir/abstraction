import os, sys
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

sys.path.append("svcca")
import cca_core
import pwcca
import models
from utils import *

parser = argparse.ArgumentParser(description='Representational Similarity from Activations')

parser.add_argument('--run-mode', '--rm', default='in-net',
                help='Running mode: {in-net, between-nets}')
parser.add_argument('--method', '-m', default='RV',
                help='Correlation method to use: mean SVCCA OR PWCCA (default: RV)')
parser.add_argument('--architecture', '-a', default='resnet18k_4',
                help='architecture (default: cnn5)')
parser.add_argument('--architecture2', '--a2', default='resnet18k_4',
                help='Second architecture (default: resnet18k_4)')
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


def _plot_helper(arr_x, arr_y, baseline_y,  xlabel, ylabel, save_path="svcca_default.png"):
    plt.plot(arr_x, arr_y, label="Layers")
    plt.plot(arr_x, baseline_y, label="Baseline")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='upper left')    
    plt.grid()
    plt.savefig(save_path)

def _plot_helper_per_epoch(arr_x, ccas, baseline, epochs, save_path="rep_similarity_default.png", in_net=True):
    plt.xlabel("Layer")
    plt.ylabel("Similarity")
    plt.grid()
    
    if epochs == None:    
        plt.plot(arr_x, ccas[0], color='b', marker="o", linestyle="-")
        plt.savefig(save_path)
        plt.clf()
        return

    c = np.array(epochs)

    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.inferno)
    cmap.set_array([])
    
    plt.plot(arr_x, baseline,  color='k', linestyle='--', label='baseline')
    for idx, epoch in enumerate(epochs):
        plt.plot(arr_x, ccas[idx], c=cmap.to_rgba(idx))

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

def _get_pwcca_baseline(acts1, acts2, num_datapoints):

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

    aa = np.dot(b1, b1.T)
    aa -= np.diag(np.diag(aa))

    bb = np.dot(b2, b2.T)
    bb -= np.diag(np.diag(bb))

    nom = np.sum(np.multiply(aa, bb))
    denum = np.sqrt( np.sum(np.multiply(aa, aa)) * np.sum(np.multiply(bb, bb)) )
    
    RV_coeff = nom / denum
    return RV_coeff


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
        acts1 = np.mean(acts1, axis=(1,2))

    if len(acts2.shape) > 2:
        # Activation is convolutional. Perform average pooling.
        acts2 = np.mean(acts2, axis=(1,2))

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
        acts1 = np.mean(acts1, axis=(1,2))

    if len(acts2.shape) > 2:
        # Activation is convolutional. Perform average pooling.
        acts2 = np.mean(acts2, axis=(1,2))

    # Get the maximum number of datapoints that is common in the activation pair.
    #print(num_datapoints)
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
        acts1 = np.mean(acts1, axis=(1,2))

    if len(acts2.shape) > 2:
        # Activation is convolutional. Perform average pooling.
        acts2 = np.mean(acts2, axis=(1,2))

    # Get the maximum number of datapoints that is common in the activation pair.
    #print(num_datapoints)
    num_datapoints = min(acts1.shape[0], acts2.shape[0], num_datapoints)
    #print("Num of samples for RV analysis:",num_datapoints)
    acts1 = acts1[:num_datapoints, :]
    acts2 = acts2[:num_datapoints, :]

    #cacts1 = acts1 - np.mean(acts1, axis=1, keepdims=True)
    #cacts2 = acts2 - np.mean(acts2, axis=1, keepdims=True)
    aa = np.dot(acts1, acts1.T)
    aa -= np.diag(np.diag(aa))

    bb = np.dot(acts2, acts2.T)
    bb -= np.diag(np.diag(bb))

    nom = np.sum(np.multiply(aa, bb))
    denum = np.sqrt( np.sum(np.multiply(aa, aa)) * np.sum(np.multiply(bb, bb)) )
    
    RV_coeff = nom / denum
    return RV_coeff

'''
def _get_mean_svcca_conv_avg_pool(acts1, acts2, top_dims=20):
    avg_acts1 = np.mean(acts1, axis=(0,1))
    avg_acts2 = np.mean(acts2, axis=(0,1))

    num_datapoints = min(avg_acts1.shape[1], avg_acts2.shape[1])

    cacts1 = avg_acts1[:, :num_datapoints] - np.mean(avg_acts1[:, :num_datapoints], axis=1, keepdims=True)
    cacts2 = avg_acts2[:, :num_datapoints] - np.mean(avg_acts2[:, :num_datapoints], axis=1, keepdims=True)

    # Perform SVD
    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)
    U2, s2, V2 = np.linalg.svd(cacts2, full_matrices=False)

    top_dims = min(s1.shape[0], s2.shape[0], top_dims)
    svacts1 = np.dot(s1[:top_dims]*np.eye(top_dims), V1[:top_dims])
    # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
    svacts2 = np.dot(s2[:top_dims]*np.eye(top_dims), V2[:top_dims])
    # can also compute as svacts1 = np.dot(U2.T[:20], cacts2)

    svcca_results = cca_core.get_cca_similarity(svacts1, svacts2, epsilon=1e-10, verbose=False)
    return np.mean(svcca_results["cca_coef1"])
'''

def _produce_activations(arch, checkpoint_epoch):
    clean_command = "rm activations/{0}/*".format(arch)
    os.system(clean_command)
    
    command = "python3 main.py -a {0} --resume checkpoints/{0}/checkpoint_{1}.tar \
    --evaluate --collect-acts -j 1 --activation-dir activations/{0}/".format(arch, checkpoint_epoch)
    os.system(command)

def _get_architecture_activations(arch):
    act_dict = defaultdict()
    all_dict = defaultdict(list)
    #conv_dict = defaultdict(list)
    #fc_dict = defaultdict(list)

    for r, d, f in os.walk(os.path.join("activations", arch)):
        for act in sorted(f):
            act_path = os.path.join(r, act)
            key = act.split("_")[0].split(".")

            if "module" in key:
                key.remove("module")

            if "downsample" in key:
                continue

            key = ".".join(key)
            all_dict[key].append(act_path)

            #if key.split(".")[0] == "features":
            #    conv_dict[key].append(act_path)
            #else:
            #    fc_dict[key].append(act_path)

    '''
    all_dict = {**conv_dict, **fc_dict}

    conv_keys = np.array(list(conv_dict.keys()))
    fc_keys = np.array(list(fc_dict.keys()))

    sorted_index = np.argsort([ int(item.split(".")[-1]) for item in conv_keys])
    conv_keys = conv_keys[sorted_index]

    sorted_index = np.argsort([ int(item.split(".")[-1]) for item in fc_keys])
    fc_keys = fc_keys[sorted_index]
    '''

    '''
    for key in fc_keys:
        #acts = [it for it in fc_dict[key]]
        #acts = np.vstack(acts)
        fc_acts.append(fc_dict[key])

    for key in conv_keys:
        #acts = [it for it in conv_dict[key]]
        acts = np.vstack(acts)
        conv_acts.append(acts)
    '''
    #layer_acts = conv_acts + fc_acts
    model = model = models.__dict__[arch]()
    
    dict_keys = []
    for item in model.named_parameters():
        splt = item[0].split(".")
        if splt[-1] == "weight":
            l = ".".join(splt[:-1])
            if l in all_dict:
                dict_keys.append(l)

    for key in dict_keys:
        act_dict[key] = np.vstack([np.load(it) for it in all_dict[key]])

    return act_dict, dict_keys


def _calculate_sim_from_activations(method, arch, num_datapoints, reference_layer, epoch, baseline=False):
    """ calculates similarity of the activations of all layers to a reference layer for a specific checkpoint """

    act_dict, dict_keys = _get_architecture_activations(arch)
    #dict_keys = np.concatenate((conv_keys, fc_keys))

    if len(dict_keys) <= reference_layer:
        raise ValueError("reference layer index is out of bound")
        return
    
    '''
    conv_cca = []
    baseline_conv = []
    for idx, l in enumerate(conv_acts):
        corr = _get_mean_svcca(l, conv_acts[-1])
        corr_baseline =  _get_mean_svcca_baseline(l, conv_acts[-1])
        conv_cca.append(corr)
        baseline_conv.append(corr_baseline)
        print("Comparing:", conv_keys[idx], conv_keys[-1])
        print("  Original avg. SVCCA: {:.2f}".format( corr ))
        print("  Baseline avg. SVCCA: {:.2f}".format(corr_baseline))
        print(" ")

    fc_cca = []
    baseline_fc = []
    for idx, l in enumerate(fc_acts):
        corr = _get_mean_svcca(l, fc_acts[-1])
        corr_baseline =  _get_mean_svcca_baseline(l, fc_acts[-1])
        fc_cca.append(corr)
        baseline_fc.append(corr_baseline)
        print("Comparing:", fc_keys[idx], fc_keys[-1])
        print("  Original avg. SVCCA: {:.2f}".format( corr ))
        print("  Baseline avg. SVCCA: {:.2f}".format( corr_baseline ))
        print(" ")
    '''

    all_cca = []
    baseline_cca = []
    all_x = []
    try:
        for l in dict_keys:
            all_x.append(dict_keys.index(l)+1) #TODO: A better naming for the layer axis.
            
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
            print("  Baseline avg. similarity: {:.2f}".format(baseline_corr))
            print(" ")


        fig_save_path = "figures/layer_similarity/{3}_{1}_refLayer{2}_{0}_epoch{4}.png".format(num_datapoints, 
                                                                            arch, reference_layer, method, epoch)
        _plot_helper_per_epoch(all_x, [all_cca,], baseline_cca, None, fig_save_path)
        return all_x, all_cca, baseline_cca
    except Exception as e:
        traceback.print_exc()
        return

def calculate_sim_per_checkpoint(method, arch, num_datapoints, reference_layer, min_epoch, max_epoch):
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
    for r, d, f in os.walk(os.path.join("checkpoints", arch)):
        epochs = np.array([int(file.split(".")[0].split("_")[-1]) for file in f])
        sort_index  = np.argsort(epochs)
        
        checkpoints = np.array([file for file in f])
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
        for idx, checkpoint_epoch in enumerate(epochs.tolist()):
            used_epochs.append(checkpoint_epoch)
            used_checkpoints.append(checkpoints[idx])

            _produce_activations(arch, checkpoint_epoch)

            layer_axis, simil, baseline = _calculate_sim_from_activations(method, arch, num_datapoints, 
                                reference_layer, checkpoint_epoch, ( idx == (epochs.shape[0]-1) ) ) 
            similarities.append(simil)

    fig_save_path = "figures/layer_similarity/{3}_{1}_refLayer{2}_{0}.png".format(num_datapoints, 
                                                                                arch, reference_layer, method)

    data_save_path = "data/layer_similarity/{3}_{1}_refLayer{2}_{0}.pickle".format(num_datapoints, 
                                                                                arch, reference_layer, method)
    _plot_helper_per_epoch(layer_axis, similarities, baseline, used_epochs, fig_save_path)

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
        x_axis.append(idx+1) #TODO: A better naming for the layer can be considered for the plots.
        
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
                                 )

    elif args.run_mode.lower() == 'between-nets':
        calculate_sim_between_nets(args.architecture,
                                    args.architecture2,
                                    args.method,
                                    args.epoch_a1,
                                    args.epoch_a2,
                                    args.num_datapoints
                                )


if __name__ == '__main__':
    main()
