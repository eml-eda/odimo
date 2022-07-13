import math
from pathlib import Path

import torch.nn as nn

from . import utils
from . import quant_module as qm
from . import hw_models as hw
from .quant_resnet import quantres8_fp_foldbn


# DJP
__all__ = [
    'mixres8_diana5', 'mixres8_diana10', 'mixres8_diana100',
]


def conv3x3(conv_func, hw_model, in_planes, out_planes, bias=False,
            stride=1, groups=1, fix_qtz=False, **kwargs):
    "3x3 convolution with padding"
    if conv_func != nn.Conv2d:
        return conv_func(hw_model, in_planes, out_planes,
                         kernel_size=3, groups=groups, stride=stride,
                         padding=1, bias=bias, fix_qtz=fix_qtz, **kwargs)
    else:
        return conv_func(in_planes, out_planes,
                         kernel_size=3, groups=groups, stride=stride,
                         padding=1, bias=bias, **kwargs)


# MR
def fc(conv_func, hw_model, in_planes, out_planes, stride=1, groups=1, search_fc=None, **kwargs):
    "fc mapped to conv"
    return conv_func(hw_model, in_planes, out_planes, kernel_size=1, groups=groups, stride=stride,
                     padding=0, bias=True, fc=search_fc, **kwargs)


# MR
class Backbone(nn.Module):
    def __init__(self, conv_func, hw_model, input_size, bn, **kwargs):
        super().__init__()
        self.bb_1 = BasicBlockGumbel(conv_func, hw_model, 16, 16, stride=1, bn=bn, **kwargs)
        self.bb_2 = BasicBlockGumbel(conv_func, hw_model, 16, 32, stride=2, bn=bn, **kwargs)
        self.bb_3 = BasicBlockGumbel(conv_func, hw_model, 32, 64, stride=2, bn=bn, **kwargs)
        self.pool = nn.AvgPool2d(kernel_size=8)

    def forward(self, x, temp, is_hard):
        x = self.bb_1(x, temp, is_hard)
        x = self.bb_2(x, temp, is_hard)
        x = self.bb_3(x, temp, is_hard)
        x = self.pool(x)
        return x


class BasicBlockGumbel(nn.Module):
    def __init__(self, conv_func, hw_model, inplanes, planes,
                 stride=1, downsample=None, bn=True, **kwargs):
        self.bn = bn
        self.use_bias = not bn
        super().__init__()
        self.conv1 = conv3x3(conv_func, hw_model, inplanes, planes,
                             stride=stride, bias=self.use_bias, **kwargs)
        if bn:
            self.bn1 = nn.BatchNorm2d(planes, affine=bn)
        self.conv2 = conv3x3(conv_func, hw_model, planes, planes,
                             bias=self.use_bias, **kwargs)
        if bn:
            self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or inplanes != planes:
            self.downsample = conv_func(hw_model, inplanes, planes,
                                        kernel_size=1, stride=stride, groups=1, bias=self.use_bias,
                                        **kwargs)
            if bn:
                self.bn_ds = nn.BatchNorm2d(planes)
        else:
            self.downsample = None

    def forward(self, x, temp, is_hard):
        if self.downsample is not None:
            residual = x
        else:
            residual = x

        out = self.conv1(x, temp, is_hard)
        if self.bn:
            out = self.bn1(out)
        out = self.conv2(out, temp, is_hard)
        if self.bn:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual, temp, is_hard)
            if self.bn:
                residual = self.bn_ds(residual)

        out += residual

        return out


class TinyMLResNet(nn.Module):
    def __init__(self, conv_func, hw_model,
                 search_fc=None, input_size=32, num_classes=10, bn=True, **kwargs):
        if 'abits' in kwargs:
            print('abits: {}'.format(kwargs['abits']))
        if 'wbits' in kwargs:
            print('wbits: {}'.format(kwargs['wbits']))

        self.inplanes = 16
        self.conv_func = conv_func
        self.hw_model = hw_model
        self.search_types = ['fixed', 'mixed', 'multi']
        if search_fc in self.search_types:
            self.search_fc = search_fc
        else:
            self.search_fc = False
        self.bn = bn
        self.use_bias = not bn
        super().__init__()
        self.gumbel = kwargs.get('gumbel', False)

        # Model
        self.conv1 = conv3x3(conv_func, hw_model, 3, 16, stride=1, groups=1,
                             bias=self.use_bias, **kwargs)
        if bn:
            self.bn1 = nn.BatchNorm2d(16)
        self.backbone = Backbone(conv_func, hw_model, input_size, bn, **kwargs)
        self.fc = fc(conv_func, hw_model, 64, num_classes, search_fc=self.search_fc, **kwargs)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, temp, is_hard):
        x = self.conv1(x, temp, is_hard)
        if self.bn:
            x = self.bn1(x)

        x = self.backbone(x, temp, is_hard)

        x = x if self.search_fc else x.view(x.size(0), -1)

        if self.search_fc:
            x = self.fc(x, temp, is_hard)
            return x[:, :, 0, 0]
        else:
            x = self.fc(x)
            return x

    def complexity_loss(self):
        loss = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                loss += m.complexity_loss()
        return loss

    def fetch_best_arch(self):
        sum_bitops, sum_bita, sum_bitw = 0, 0, 0
        sum_mixbitops, sum_mixbita, sum_mixbitw = 0, 0, 0
        layer_idx = 0
        best_arch = None
        for m in self.modules():
            if isinstance(m, self.conv_func):
                outs = m.fetch_best_arch(layer_idx)  # Return tuple
                layer_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = outs
                if best_arch is None:
                    best_arch = layer_arch
                else:
                    for key in layer_arch.keys():
                        if key not in best_arch:
                            best_arch[key] = layer_arch[key]
                        else:
                            best_arch[key].append(layer_arch[key][0])
                sum_bitops += bitops
                sum_bita += bita
                sum_bitw += bitw
                sum_mixbitops += mixbitops
                sum_mixbita += mixbita
                sum_mixbitw += mixbitw
                layer_idx += 1
        return best_arch, sum_bitops, sum_bita, sum_bitw, sum_mixbitops, sum_mixbita, sum_mixbitw


# MR
def mixres8_diana5(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    # NB: 2 bits is equivalent for ternary weights!!
    search_model = TinyMLResNet(
        qm.MultiPrecActivConv2d, hw.diana(analog_speedup=5.),
        search_fc='multi', wbits=[2, 8], abits=[8], bn=False,
        share_weight=True, **kwargs)

    # Get folded pretrained model
    folded_fp_model = quantres8_fp_foldbn(arch_cfg_path)
    folded_state_dict = folded_fp_model.state_dict()

    # Delete folded model
    del folded_fp_model

    # Translate folded state dict in a format compatible with searchable layers
    search_state_dict = utils.fpfold_to_q(folded_state_dict)
    search_model.load_state_dict(search_state_dict, strict=False)

    # Init quantization scale param
    utils.init_scale_param(search_model)

    return search_model


# MR
def mixres8_diana10(**kwargs):
    # NB: 2 bits is equivalent for ternary weights!!
    return TinyMLResNet(
        qm.MultiPrecActivConv2d, hw.diana(analog_speedup=10.),
        search_fc='multi', wbits=[2, 8], abits=[8],
        share_weight=True, **kwargs)


# MR
def mixres8_diana100(**kwargs):
    # NB: 2 bits is equivalent for ternary weights!!
    return TinyMLResNet(
        qm.MultiPrecActivConv2d, hw.diana(analog_speedup=100.),
        search_fc='multi', wbits=[2, 8], abits=[8],
        share_weight=True, **kwargs)
