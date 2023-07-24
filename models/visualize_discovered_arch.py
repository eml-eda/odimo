import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


def visualize_discovered_arch(path):
    path = Path(path)
    model_name = str(path.parent)
    if not path.exists():
        raise ValueError(f'{path} does not exists!')
    state_dict = torch.load(path)['state_dict']
    alpha_state_dict = dict()
    prec_assign = dict()
    idx = 0
    for name, params in state_dict.items():
        name_list = name.split('.')
        suffix = name_list[-1]
        layer_name = '.'.join(name_list[:-2])
        if suffix == 'alpha_weight':
            params_np = params.detach().cpu().numpy()
            alpha_state_dict[layer_name] = params_np
            prec = params_np.argmax(axis=0)
            ch = len(prec)
            frac_digital = sum(prec == 0) / ch
            frac_analog = sum(prec == 1) / ch
            prec_assign[idx] = {
                'layer': layer_name,
                'analog': frac_analog,
                'digital': frac_digital,
                }
            idx += 1
    n_layer = idx
    figsize = [2*n_layer, n_layer]
    fig, ax = plt.subplots(figsize=figsize)
    df = pd.DataFrame(prec_assign).T
    # import pdb; pdb.set_trace()
    for i in reversed(range(n_layer)):
        ax.barh(df.iloc[i]['layer'], df.iloc[i]['analog'],
                color='#ff595e', edgecolor='white', label='analog')
        ax.barh(df.iloc[i]['layer'], df.iloc[i]['digital'], left=df.iloc[i]['analog'],
                color='#1982c4', edgecolor='white', label='digital')
    plt.yticks(fontsize=16, rotation=45)
    plt.xticks(fontsize=18, rotation=45)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], ncol=2, loc=[0.5, 1], fontsize=20)
    fig.set_tight_layout(True)
    fig.savefig(model_name + '/arch.png')
    print(f'Layer-wise cycles profile saved @ {model_name}', end='\n')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Discovered Arch')
    parser.add_argument('path', type=str, help='path to discovered network')
    args = parser.parse_args()

    visualize_discovered_arch(args.path)
