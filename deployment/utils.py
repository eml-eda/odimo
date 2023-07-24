from enum import Enum, auto
import torch.nn as nn

__all__ = [
    'add_attributes',
    'IntegerizationMode'
]


def add_attributes(ref_mod: nn.Module, target_mod: nn.Module,
                   exclude_attr: tuple = ()):
    for attr in dir(ref_mod):
        if attr not in dir(target_mod) and not isinstance(getattr(ref_mod, attr), exclude_attr):
            setattr(target_mod, attr, getattr(ref_mod, attr))


class IntegerizationMode(Enum):
    Int = auto()
    FakeInt = auto()
