import copy

import torch
import torch.fx as fx
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from . import quant_module as qm


def _non_zero_frac(w, init_scale_param, cout):
    dim = w.dim()
    axs = tuple(i for i in range(1, dim))
    scaled_w = w / torch.exp(init_scale_param).view(cout, 1, 1, 1)
    ch_numel = w.numel() / cout  # N. of elements for each output channel slice of w
    if cout != 1:
        frac = scaled_w.clamp_(-1, 1).round().count_nonzero(dim=axs) / ch_numel
    else:
        frac = scaled_w.clamp_(-1, 1).round().count_nonzero() / ch_numel
        frac = frac.unsqueeze(0)
    return frac


def _parent_name(target):
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name


# Works for length 2 patterns with 2 modules
def _matches_module_pattern(pattern, node, modules):
    if len(node.args) == 0:
        return False
    nodes = (node.args[0], node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != 'call_module':
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not expected_type:
            return False
    return True


def _replace_node_module(node, modules, new_module):
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)


def adapt_scale_params(state_dict, model):
    new_dict = copy.deepcopy(state_dict)
    for key in state_dict.keys():
        if 'scale_param' in key:
            dim = model.state_dict()[key].shape
            new_dict[key] = state_dict[key].repeat(dim)

    return new_dict


# http://tinyurl.com/2p9a22kd <- copied from torch.fx experimental (torch v11.0)
def fold_bn(model, inplace=False):
    patterns = [(nn.Conv1d, nn.BatchNorm1d),
                (nn.Conv2d, nn.BatchNorm2d),
                (nn.Conv3d, nn.BatchNorm3d)]
    if not inplace:
        model = copy.deepcopy(model)
    fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    new_graph = copy.deepcopy(fx_model.graph)

    for pattern in patterns:
        for node in new_graph.nodes:
            if _matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                conv = modules[node.args[0].target]
                bn = modules[node.target]
                if not bn.track_running_stats:
                    continue
                fused_conv = fuse_conv_bn_eval(conv, bn)
                _replace_node_module(node.args[0], modules, fused_conv)
                node.replace_all_uses_with(node.args[0])
                new_graph.erase_node(node)
    return fx.GraphModule(fx_model, new_graph)


def fp_to_q(state_dict):
    state_dict = copy.deepcopy(state_dict)
    converted_dict = dict()

    for name, params in state_dict.items():
        full_name = name
        name = name.split('.')[-1]
        if name in ['weight']:
            name_list = full_name.split('.')
            name_list.insert(-2, 'mix_weight')
            new_name = '.'.join(name_list)
            converted_dict[new_name] = params

    return converted_dict


def fpfold_to_q(state_dict):
    state_dict = copy.deepcopy(state_dict)
    converted_dict = dict()

    for name, params in state_dict.items():
        full_name = name
        name = name.split('.')[-1]
        if name in ['weight', 'bias']:
            name_list = full_name.split('.')
            name_list.insert(-2, 'mix_weight')
            new_name = '.'.join(name_list)
            converted_dict[new_name] = params

    return converted_dict


def init_scale_param(model):
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, qm.QuantMultiPrecConv2d):
                w = module.conv.weight
                for submodule in module.mix_weight:
                    nb = submodule.num_bits
                    if nb == 2:
                        continue
                        cout = w.shape[0]  # Per-Ch scale factor
                        # cout = 1  # Per-Layer scale factor
                        # Init scale param to have ~50% of weights != 0
                        # init_scale_param = torch.zeros([cout], dtype=torch.float32)
                        init_scale_param = submodule.scale_param
                        delta = .1
                        target = .5
                        non_zero_frac = _non_zero_frac(w, init_scale_param, cout)
                        while any(non_zero_frac < target):
                            init_scale_param[non_zero_frac < target] -= delta
                            non_zero_frac = _non_zero_frac(w, init_scale_param, cout)
                    else:
                        # Init scale param to maximize the quantization range
                        init_scale_param = torch.log(2 * w.abs().max())
                        # init_scale_param = torch.log(w.abs().max())
                        submodule.scale_param.data = init_scale_param


def q_to_fp(state_dict):
    converted_dict = copy.deepcopy(state_dict)

    for name, params in state_dict.items():
        full_name = name
        name = '.'.join(name.split('.')[-2:])
        if name in ['conv.weight', 'conv.bias']:
            name_list = full_name.split('.')
            name_list.remove('mix_weight')
            new_name = '.'.join(name_list)
            converted_dict[new_name] = params

    return converted_dict
