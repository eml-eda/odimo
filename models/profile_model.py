import argparse
from pathlib import Path
import pickle

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


def profile_cycles(arch,
                   input_shape='default',
                   pretrained_arch=None,
                   plot_layer_detail=False,
                   plot_net_graph=False,
                   save_pkl=False):
    model = _ARCH_FUNC[arch](None)
    if input_shape == 'default':
        dummy_input = torch.randn(_INP_SHAPE[arch])
    else:
        img_size = tuple(input_shape, input_shape)
        inp_shape = (1, 3) + img_size
        dummy_input = torch.randn(inp_shape)
    gm = fx.symbolic_trace(model)
    modules = dict(gm.named_modules())
    arch_details = dict()

    if pretrained_arch is not None:
        path = Path(pretrained_arch)
        if not path.exists():
            raise ValueError(f'{path} does not exists!')
        state_dict = torch.load(path)['state_dict']

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
                arch_details[name]['analog_func'] = np.array([
                    analog_cycles(ch_in, ch, k_x, k_y, out_x, out_y)[1]
                    for ch in range(1, ch_out+1)])
                arch_details[name]['analog_latency'] = \
                    arch_details[name]['analog_func'].max()
                arch_details[name]['digital_func'] = np.array([
                    digital_cycles(ch_in, ch, k_x, k_y, out_x, out_y)[1]
                    for ch in range(1, ch_out+1)])
                arch_details[name]['digital_latency'] = \
                    arch_details[name]['digital_func'].max()
                arch_details[name]['min_latency'] = \
                    (np.flip(arch_details[name]['analog_func']) -
                     arch_details[name]['digital_func'] > 0)
                alpha = state_dict[f'{name}.mix_weight.alpha_weight'].detach().cpu().numpy()
                prec = alpha.argmax(axis=0)
                ch_d = sum(prec == 0)
                ch_a = sum(prec == 1)
                _, nas_d = digital_cycles(ch_in, ch_d, k_x, k_y, out_x, out_y)
                _, nas_a = analog_cycles(ch_in, ch_a, k_x, k_y, out_x, out_y)
                nas_max = max(nas_d, nas_a)
                arch_details[name]['NAS_digital'] = nas_d
                arch_details[name]['NAS_analog'] = nas_a
                arch_details[name]['NAS_max'] = nas_max

    if plot_layer_detail:
        df = pd.DataFrame(arch_details)
        n_layer = len(df.columns)
        # figsize = [1*x for x in plt.rcParams["figure.figsize"]]
        figsize = [n_layer, 2*n_layer]
        fig, axis = plt.subplots(n_layer, figsize=figsize)

        for idx, col in enumerate(df):
            axis[idx].plot(df[col]['x_ch'], df[col]['analog_func'],
                           color='#ff595e', label='analog')
            axis[idx].plot(df[col]['x_ch'], df[col]['digital_func'],
                           color='#1982c4', label='digital')
            axis[idx].set_title(col)

        handles, labels = axis[-1].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2)
        fig.set_tight_layout(True)
        fig.savefig(f'{str(arch)}_cycles.png')
        print(f'Layer-wise cycles profile saved @ {str(arch)}_cycles.png', end='\n')

    if save_pkl:
        with open(f'details_{arch}.pickle', 'wb') as h:
            pickle.dump(arch_details, h, protocol=pickle.HIGHEST_PROTOCOL)

    return arch_details


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Model')
    parser.add_argument('arch', type=str, help='Architecture name')
    parser.add_argument('--path', type=str, default=None, help='path to discovered network')
    parser.add_argument('--plot-net-graph', action='store_true', default=False,
                        help='Architecture name')
    parser.add_argument('--plot-layer-detail', action='store_true', default=False,
                        help='Architecture name')
    parser.add_argument('--save-pkl', action='store_true', default=False,
                        help='Architecture name')
    args = parser.parse_args()

    if args.arch not in _ARCH_FUNC:
        raise ValueError(
            f'''{args.arch} is not supported. List of supported models: {_ARCH_FUNC.keys()}''')

    profile_cycles(args.arch,
                   pretrained_arch=args.path,
                   plot_layer_detail=args.plot_layer_detail,
                   plot_net_graph=args.plot_net_graph,
                   save_pkl=args.save_pkl)
