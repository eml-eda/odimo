import torch.nn as nn

__all__ = [
    'add_attributes',
]


def add_attributes(ref_mod: nn.Module, target_mod: nn.Module):
    for attr in dir(ref_mod):
        if attr not in dir(target_mod):
            setattr(target_mod, attr, getattr(ref_mod, attr))
