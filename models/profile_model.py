import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def profile_cycles(arch, plot_net_graph=False):
    model = _ARCH_FUNC[arch](None)
    dummy_input = torch.randn(_INP_SHAPE[arch])
    gm = fx.symbolic_trace(model)
    modules = dict(gm.named_modules())
    arch_details = dict()

    if plot_net_graph:
        gd = FxGraphDrawer(gm, str(arch))
        gd.get_dot_graph().write_png(f'{str(arch)}_graph.png')
        print(f'Graph Plot saved @ {str(arch)}_graph.png', end='\n')

    ShapeProp(gm).propagate(dummy_input)

    # Build dict with layers' shapes and cycles
    for node in gm.graph.nodes:
        if node.target in modules.keys():
            if isinstance(modules[node.target], nn.Conv2d):
                name = '.'.join(node.target.split('.')[:-1])
                conv = modules[node.target]
                out_shape = node.meta['tensor_meta'].shape
                ch_in = conv.in_channels
                ch_out = conv.out_channels
                k_x = conv.kernel_size[0]
                k_y = conv.kernel_size[1]
                out_x = out_shape[-2]
                out_y = out_shape[-1]
                arch_details[name] = {
                    'ch_in': ch_in,
                    'ch_out': ch_out,
                    'k_x': k_x,
                    'k_y': k_y,
                    'out_x': out_x,
                    'out_y': out_y,
                }
                arch_details[name]['x_ch'] = np.arange(1, ch_out+1)
                arch_details[name]['a_cycles'] = np.array([
                    analog_cycles(ch_in, ch, k_x, k_y, out_x, out_y)[1]
                    for ch in range(1, ch_out+1)])
                arch_details[name]['d_cycles'] = np.array([
                    digital_cycles(ch_in, ch, k_x, k_y, out_x, out_y)[1]
                    for ch in range(1, ch_out+1)])

    df = pd.DataFrame(arch_details)
    n_layer = len(df.columns)
    # figsize = [1*x for x in plt.rcParams["figure.figsize"]]
    figsize = [n_layer, 2*n_layer]
    fig, axis = plt.subplots(n_layer, figsize=figsize)

    for idx, col in enumerate(df):
        axis[idx].plot(df[col]['x_ch'], df[col]['a_cycles'], color='#ff595e', label='analog')
        axis[idx].plot(df[col]['x_ch'], df[col]['d_cycles'], color='#1982c4', label='digital')
        axis[idx].set_title(col)

    handles, labels = axis[-1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2)
    fig.set_tight_layout(True)
    fig.savefig(f'{str(arch)}_cycles.png')
    print(f'Layer-wise cycles profile saved @ {str(arch)}_cycles.png', end='\n')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Model')
    parser.add_argument('arch', type=str, help='Architecture name')
    parser.add_argument('--plot-net-graph', action='store_true', default=False,
                        help='Architecture name')
    args = parser.parse_args()

    if args.arch not in _ARCH_FUNC:
        raise ValueError(
            f'''{args.arch} is not supported. List of supported models: {_ARCH_FUNC.keys()}''')

    profile_cycles(args.arch, plot_net_graph=args.plot_net_graph)
