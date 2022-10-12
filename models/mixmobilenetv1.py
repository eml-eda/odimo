from pathlib import Path
from turtle import forward

import numpy as np
import torch
import torch.nn as nn

from . import utils
from . import quant_module as qm
from . import hw_models as hw
from .quant_mobilenetv1 import quantmobilenetv1_fp, quantmobilenetv1_fp_foldbn

# MR
__all__ = [
    'mixmobilenetv1_diana_full',
    'mixmobilenetv1_diana_reduced',
]


class BasicBlockGumbel(nn.Module):

    def __init__(self):
        ...

    def forward(self, x):
        ...


class Backbone(nn.Module):

    def __init__(self):
        ...

    def forward(self, x):
        ...

class MobileNetV1(nn.Module):

    def __init__(self):
        ...

    def forward(self, x):
        ...

    def complexity_loss(self):
        ...

    def fetch_best_arch(self):
        ...


def mixmobilenetv1_diana_full(arch_cfg_path, **kwargs):
    search_model = MobileNetV1(
        qm.MultiPrecActivConv2d, hw.diana(), [True]*28,
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, **kwargs
    )
    return _mixmobilenetv1_diana(arch_cfg_path, search_model)


def mixmobilenetv1_diana_reduced(arch_cfg_path, **kwargs):
    is_searchable = utils.detect_ad_tradeoff(
        quantmobilenetv1_fp(None, pretrained=False),
        torch.rand((1, 3, 64, 64)))
    search_model = MobileNetV1(
        qm.MultiPrecActivConv2d, hw.diana(), is_searchable,
        search_fc='multi', wbits=[8, 2], abits=[7], bn=False,
        share_weight=True, **kwargs
    )
    return _mixmobilenetv1_diana(arch_cfg_path, search_model)


def _mixmobilenetv1_diana(arch_cfg_path, search_model):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    # Get folded pretrained model
    folded_fp_model = quantmobilenetv1_fp_foldbn(arch_cfg_path)
    folded_state_dict = folded_fp_model.state_dict()

    # Delete folded model
    del folded_fp_model

    # Translate folded state dict in a format compatible with searchable layers
    search_state_dict = utils.fpfold_to_q(folded_state_dict)
    search_model.load_state_dict(search_state_dict, strict=False)

    # Init quantization scale param
    utils.init_scale_param(search_model)

    return search_model
