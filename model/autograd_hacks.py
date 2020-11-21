"""
Library for extracting interesting quantites from autograd, see README.md

Not thread-safe because of module-level variables

Notation:
o: number of output classes (exact Hessian), number of Hessian samples (sampled Hessian)
n: batch-size
do: output dimension (output channels for convolution)
di: input dimension (input channels for convolution)
Hi: per-example Hessian of matmul, shaped as matrix of [dim, dim], indices have been row-vectorized
Hi_bias: per-example Hessian of bias
Oh, Ow: output height, output width (convolution)
Kh, Kw: kernel height, kernel width (convolution)

Jb: batch output Jacobian of matmul, output sensitivity for example,class pair, [o, n, ....]
Jb_bias: as above, but for bias

A, activations: inputs into current layer
B, backprops: backprop values (aka Lop aka Jacobian-vector product) observed at current layer

"""
import os
from collections import defaultdict
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

_supported_layers = ['Linear', 'Conv2d']  # Supported layer class types
#_activation_layers = ["ReLU"]
_hooks_disabled: bool = False           # work-around for https://github.com/pytorch/pytorch/issues/25723

_forward_handles = []
_backward_handles = []

_activations = {}
_backprops = {}

def add_hooks(model: nn.Module, grad1: bool = True) -> None:
    """
    Adds hooks to model to save activations and backprop values.

    The hooks will
    1. save activations into param.activations during forward pass
    2. append backprops to params.backprops_list during backward pass.

    Call "remove_hooks(model)" to disable this.

    Args:
        model:
    """

    global _hooks_disabled, _forward_handles, _backward_handles
    _hooks_disabled = False

    (last_name, last_layer) = list(model.named_modules())[-1]
    for name, layer in model.named_modules():
        if grad1:
            if is_module_supported(name, layer):
                # partial to assign the layer name to each hook
                _forward_handles.append(layer.register_forward_hook(partial(_capture_activations_before_layer, name)))
                _backward_handles.append(layer.register_backward_hook(partial(_capture_backprops, name)))

        else:
            if is_module_supported_activation(name, layer):
                #print("HOOK installed on", name)
                _forward_handles.append(layer.register_forward_hook(partial(_capture_activations_after_layer, name)))
            if (name, layer) == (last_name, last_layer):
                #print("HOOK installed on", name)
                _forward_handles.append(layer.register_forward_hook(partial(_capture_final_activations, name)))


def remove_hooks(model: nn.Module) -> None:
    """
    Remove hooks added by add_hooks(model)
    """

    #assert model == 0, "not working, remove this after fix to https://github.com/pytorch/pytorch/issues/25723"

    #if not hasattr(model, 'autograd_hacks_hooks'):
    #    print("Warning, asked to remove hooks, but no hooks found")

    #else:
    for handle in _forward_handles: #model.autograd_hacks_hooks:
        handle.remove()

    for handle in _backward_handles:
        handle.remove()

    _activations.clear()
    _backprops.clear()
    #del model.autograd_hacks_hooks

def disable_hooks() -> None:
    """
    Globally disable all hooks installed by this library.
    """
    global _hooks_disabled
    _hooks_disabled = True


def enable_hooks() -> None:
    """the opposite of disable_hooks()"""

    global _hooks_disabled
    _hooks_disabled = False


def is_supported(layer: nn.Module) -> bool:
    """Check if this layer is supported"""
    return _layer_type(layer) in _supported_layers

def is_supported_activation(layer: nn.Module) -> bool:
    """Check if this layer is supported"""
    return isinstance(layer, nn.ReLU)


def is_module_supported(name, layer):
    """Check if this layer is included"""

    return is_supported(layer)

def is_module_supported_activation(name, layer):
    """Check if this layer is included"""

    return is_supported_activation(layer)


def _layer_type(layer: nn.Module) -> str:
    return layer.__class__.__name__

def _capture_activations_before_layer(name, layer: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
    """Save activations into layer.activations in forward pass"""

    if _hooks_disabled:
        return
    assert _layer_type(layer) in _supported_layers, "Hook installed on unsupported layer, this shouldn't happen"
    
    #setattr(layer, "activations", input[0].detach().cpu())
    _activations[name] = input[0].detach().cpu() #output.detach().cpu()

def _capture_activations_after_layer(name, layer: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
    """Save activations into layer.activations in forward pass"""

    if _hooks_disabled:
        return
    #assert _layer_type(layer) in _activation_layers, "Hook installed on unsupported layer, this shouldn't happen"
    assert isinstance(layer, nn.ReLU), "Hook installed on unsupported layer, this shouldn't happen"
    
    #setattr(layer, "activations", input[0].detach().cpu())
    _activations[name] = output.detach().cpu() #output.detach().cpu()

def _capture_final_activations(name, layer: nn.Module, input: List[torch.Tensor], output: torch.Tensor):
    """Save activations into layer.activations in forward pass"""

    if _hooks_disabled:
        return
    
    assert _layer_type(layer) == "Linear", "Hook installed on unsupported layer, this shouldn't happen"
    
    #setattr(layer, "activations", input[0].detach().cpu())
    _activations[name] = F.softmax(output, dim=1).detach().cpu() #output.detach().cpu()

def _capture_backprops(name, layer: nn.Module, _input, output):
    """Append backprop to layer.backprops_list in backward pass."""

    if _hooks_disabled:
        return

    _backprops[name] = output[0].detach().cpu()

def save_activations(act_dir: str, i):
    for k,v in _activations.items():
        dest = os.path.join(act_dir, "{0}_{1}.npy".format(k, i))
        np.save(dest, v.numpy())

def get_activations():
    global _activations

    return _activations

def compute_grad1(model: nn.Module, loss_type: str = 'mean') -> None:
    """
    Compute per-example gradients and save them under 'param.grad1'. Must be called after loss.backprop()

    Args:
        model:
        loss_type: either "mean" or "sum" depending whether backpropped loss was averaged or summed over batch
    """

    if _hooks_disabled:
        return

    assert len(_backprops), "No backprops detected, activate backward handles by add_hooks(model, grad1=True)"
    assert loss_type in ('sum', 'mean')
    for name, layer in model.named_modules():
        layer_type = _layer_type(layer)
        
        if not is_module_supported(name, layer):
            continue

        assert len(_activations), "No activations detected, run forward after add_hooks(model)"
        assert len(_backprops), "No backprops detected, activate backward handles by add_hooks(model, grad1=True)"

        A = _activations[name] #layer.activations
        n = A.shape[0]
        
        if loss_type == 'mean':
            B =  _backprops[name] * n #layer.backprops_list[0] * n
        else:  # loss_type == 'sum':
            B = _backprops[name] #layer.backprops_list[0]

        if layer_type == 'Linear':
            setattr(layer.weight, 'grad1', torch.einsum('ni,nj->nij', B, A))
            if layer.bias is not None:
                setattr(layer.bias, 'grad1', B)

        elif layer_type == 'Conv2d':
            A = torch.nn.functional.unfold(A, kernel_size=layer.kernel_size,\
                                        dilation=layer.dilation, padding=layer.padding, stride=layer.stride)

            B = B.reshape(n, -1, A.shape[-1])
            grad1 = torch.einsum('ijk,ilk->ijl', B, A)
            shape = [n] + list(layer.weight.shape)

            setattr(layer.weight, 'grad1', grad1.reshape(shape))
            if layer.bias is not None:
                setattr(layer.bias, 'grad1', torch.sum(B, dim=2))
