from typing import Type

import torch
import torch.nn as nn
import torch.fx as fx

from deployment.observer import ObserverBase

__all__ = [
    'build_qgraph',
]


def _compute_int_bparams(s_w, s_x, s_y, b_16, n_sh):
    b_8 = None
    n_b = None
    return b_8, n_b


def _compute_int_wparams(s_w, s_x, s_y, max_act):
    alpha = None
    n_sh = None
    return alpha, n_sh


def _extract_qinfo(module):
    s_x = None
    s_w = None
    b_16 = None
    return s_x, s_w, b_16


class QuantizationTracer(fx.Tracer):
    """Consider layers contained in `target_layers` as leaf modules.

    :param target_layers: modules that should be considered as a leaf
    :type target_layers: tuple[Type[nn.Module]
    """

    def __init__(self, target_layers: tuple[Type[nn.Module], ...]):
        super().__init__()
        self.target_layers = target_layers

    def is_leaf_module(
        self,
        m: nn.Module,
        module_qualified_name: str
    ) -> bool:
        if isinstance(m, self.target_layers):
            return True
        elif isinstance(m, ObserverBase):
            return True
        else:
            return m.__module__.startswith('torch.nn') and \
                not isinstance(m, torch.nn.Sequential)


# MR: questa funzione si puo' specializzare poi per diversi backend
def build_qgraph(
    model: nn.Module,
    output_classes: int,
    target_layers: tuple[Type[nn.Module], ...]
) -> nn.Module:
    """
    Performs the following steps traversing from output to input:
        1. Integerize weights and biases
        2. Propagate scale factors and compute integer quantization params and
        annotate nodes.
        3. Convert fake-quantized layer with integer counterparts

    :param model: nn.Module whit quantization information
    :type model: nn.Module
    :param output_classes: number of output classes
    :type output_classes: int
    :param target_layers: set of nn.Module where quantization information
    should be extracted
    :type target_layers: tuple[Type[nn.Module], ...]
    :return: a `model` copy with annotated quantization information
    :rtype: nn.Module
    """
    tracer = QuantizationTracer(target_layers)
    graph = tracer.trace(model.eval())
    name = model.__class__.__name__
    mod = fx.GraphModule(tracer.root, graph, name)
    modules = dict(mod.named_modules())
    # modules = {}
    # for name, module in mod.named_modules():
    #     modules[name] = module
    #     if isinstance(module, target_layers):
    #         continue
    #         module.integerize()  # TODO: To be Implemented

    s_y = torch.ones((output_classes,))
    for n in reversed(mod.graph.nodes):
        m = modules.get(n.target)
        if isinstance(m, target_layers):
            s_x, s_w, b_16 = _extract_qinfo(m)
            max_act = modules.get(n.next.target).max_val
            alpha, n_sh = _compute_int_wparams(s_w, s_x, s_y, max_act)
            b_8, n_b = _compute_int_bparams(s_w, s_x, s_y, b_16, n_sh)
            m.autoconvert(alpha, n_sh, b_8, n_b)  # TODO: To be implemented
            s_y = s_x  # propagate s_x backward
    mod.graph.lint()
    mod.recompile()
    return mod
