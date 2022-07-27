import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.passes.graph_drawer import FxGraphDrawer

from models.model_diana import analog_cycles, digital_cycles
from models.quant_resnet import quantres8_fp, quantres20_fp

_ARCH_FUNC = {
    'resnet8': quantres8_fp,
    'resnet20': quantres20_fp,
}

_INP_SHAPE = {
    'resnet8': (1, 3, 32, 32),
    'resnet20': (1, 3, 32, 32),
}


def profile(arch, plot_net_graph=False):
    model = _ARCH_FUNC[arch](None)
    dummy_input = torch.randn(_INP_SHAPE[arch])
    gm = fx.symbolic_trace(model)
    modules = dict(gm.named_modules())
    arch_details = dict()

    if plot_net_graph:
        gd = FxGraphDrawer(gm, str(arch))
        gd.get_dot_graph().write_png(f'{str(arch)}_graph.png')

    ShapeProp(gm).propagate(dummy_input)

    for node in gm.graph.nodes:
        if node.target in modules.keys():
            if isinstance(modules[node.target], nn.Conv2d):
                name = '.'.join(node.target.split('.')[:-1])
                conv = modules[node.target]
                out_shape = node.meta['tensor_meta'].shape
                arch_details[name] = {
                    'ch_in': conv.in_channels,
                    'ch_out': conv.out_channels,
                    'k_x': conv.kernel_size[0],
                    'k_y': conv.kernel_size[1],
                    'out_x': out_shape[-2],
                    'out_y': out_shape[-1],
                }

    a=1
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Model')
    parser.add_argument('arch', type=str, help='Architecture name')
    parser.add_argument('--plot-net-graph', type=bool, action='store_true', default=False,
                        help='Architecture name')
    args = parser.parse_args()

    if args.arch not in _ARCH_FUNC:
        raise ValueError(
            f'''{args.arch} is not supported. List of supported models: {_ARCH_FUNC.keys()}''')

    profile(args.arch, plot_net_graph=args.plot_net_graph)
