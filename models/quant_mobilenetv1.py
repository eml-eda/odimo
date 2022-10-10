# import copy
# import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import utils
from . import quant_module as qm
from . import hw_models as hw

# MR
__all__ = [
    'quantmobilenetv1_fp', 'quantmobilenetv1_fp_foldbn',
    'quantmobilenetv1_w8a7_foldbn',
    'quantmobilenetv1_w2a7_foldbn',
    'quantmobilenetv1_w2a7_true_foldbn',
    'quantmobilenetv1_minlat_foldbn',
    'quantmobilenetv1_minlat_max8_foldbn',
    ]


def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class BasicBlock(nn.Module):

    def __init__(self, conv_func, inp, oup, archws, archas,
                 stride=1, bn=True, **kwargs):
        self.bn = bn
        self.use_bias = not bn
        super().__init__()
        self.depth = conv_func(inp, inp, archws[0], archas[0],
                               kernel_size=3, stride=stride, padding=1,
                               bias=self.use_bias, groups=inp, **kwargs)
        if bn:
            self.bn_depth = nn.BatchNorm2d(inp)
        self.point = conv_func(inp, oup, archws[1], archas[1],
                               kernel_size=1, stride=1, padding=0,
                               bias=self.use_bias, groups=1, **kwargs)
        if bn:
            self.bn_point = nn.BatchNorm2d(oup)

    def forward(self, x):
        x = self.depth(x)
        if self.bn:
            x = self.bn_depth(x)
        x = self.point(x)
        if self.bn:
            x = self.bn_point(x)

        return x


class Backbone(nn.Module):

    def __init__(self, conv_func, input_size, bn, width_mult, abits, wbits,
                 **kwargs):
        super().__init__()
        self.bb_1 = BasicBlock(
            conv_func, make_divisible(32*width_mult), make_divisible(64*width_mult),
            wbits[1:3], abits[1:3], stride=1, bn=bn, **kwargs)
        self.bb_2 = BasicBlock(
            conv_func, make_divisible(64*width_mult), make_divisible(128*width_mult),
            wbits[3:5], abits[3:5], stride=2, bn=bn, **kwargs)
        self.bb_3 = BasicBlock(
            conv_func, make_divisible(128*width_mult), make_divisible(128*width_mult),
            wbits[5:7], abits[5:7], stride=1, bn=bn, **kwargs)
        self.bb_4 = BasicBlock(
            conv_func, make_divisible(128*width_mult), make_divisible(256*width_mult),
            wbits[7:9], abits[7:9], stride=2, bn=bn, **kwargs)
        self.bb_5 = BasicBlock(
            conv_func, make_divisible(256*width_mult), make_divisible(256*width_mult),
            wbits[9:11], abits[9:11], stride=1, bn=bn, **kwargs)
        self.bb_6 = BasicBlock(
            conv_func, make_divisible(256*width_mult), make_divisible(512*width_mult),
            wbits[11:13], abits[11:13], stride=2, bn=bn, **kwargs)
        self.bb_7 = BasicBlock(
            conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult),
            wbits[13:15], abits[13:15], stride=1, bn=bn, **kwargs)
        self.bb_8 = BasicBlock(
            conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult),
            wbits[15:17], abits[15:17], stride=1, bn=bn, **kwargs)
        self.bb_9 = BasicBlock(
            conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult),
            wbits[17:19], abits[17:19], stride=1, bn=bn, **kwargs)
        self.bb_10 = BasicBlock(
            conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult),
            wbits[19:21], abits[19:21], stride=1, bn=bn, **kwargs)
        self.bb_11 = BasicBlock(
            conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult),
            wbits[21:23], abits[21:23], stride=1, bn=bn, **kwargs)
        self.bb_12 = BasicBlock(
            conv_func, make_divisible(512*width_mult), make_divisible(1024*width_mult),
            wbits[23:25], abits[23:25], stride=2, bn=bn, **kwargs)
        self.bb_13 = BasicBlock(
            conv_func, make_divisible(1024*width_mult), make_divisible(1024*width_mult),
            wbits[25:27], abits[25:27], stride=1, bn=bn, **kwargs)
        self.pool = nn.AvgPool2d(int(input_size / (2**5)))

    def forward(self, x):
        out = self.bb_1(x)
        out = self.bb_2(out)
        out = self.bb_3(out)
        out = self.bb_4(out)
        out = self.bb_5(out)
        out = self.bb_6(out)
        out = self.bb_7(out)
        out = self.bb_8(out)
        out = self.bb_9(out)
        out = self.bb_10(out)
        out = self.bb_11(out)
        out = self.bb_12(out)
        out = self.bb_13(out)
        out = self.pool(out)
        return out


class MobileNetV1(nn.Module):

    def __init__(self, conv_func, hw_model, archws, archas,
                 qtz_fc=None, width_mult=.25,
                 input_size=96, num_classes=200, bn=True, **kwargs):
        print('archas: {}'.format(archas))
        print('archws: {}'.format(archws))

        self.conv_func = conv_func
        self.hw_model = hw_model
        self.search_types = ['fixed', 'mixed', 'multi']
        if qtz_fc in self.search_types:
            self.qtz_fc = qtz_fc
        else:
            self.qtz_fc = False
        self.width_mult = width_mult
        self.bn = bn
        self.use_bias = not bn
        super().__init__()

        # Model
        self.input_layer = conv_func(3, make_divisible(32*width_mult),
                                     abits=archas[0], wbits=archws[0],
                                     kernel_size=3, stride=2, padding=1,
                                     bias=False, groups=1, **kwargs)
        if bn:
            self.bn = nn.BatchNorm2d(make_divisible(32*width_mult))
        self.backbone = Backbone(conv_func, input_size, bn, width_mult,
                                 abits=archas[1:-1], wbits=archws[1:-1],
                                 **kwargs)

        # Initialize bn and conv weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

        # Final classifier
        self.fc = conv_func(
            make_divisible(1024*width_mult), num_classes,
            abits=archas[-1], wbits=archws[-1],
            kernel_size=1, stride=1, padding=0, bias=True, groups=1,
            fc=self.qtz_fc, **kwargs)

    def forward(self, x):
        x = self.input_layer(x)
        if self.bn:
            x = self.bn(x)
        x = self.backbone(x)
        x = self.fc(x)[:, :, 0, 0]

        return x

    def fetch_arch_info(self):
        sum_cycles, sum_bita, sum_bitw = 0, 0, 0
        layer_idx = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                size_product = m.size_product.item()
                memory_size = m.memory_size.item()
                wbit = m.wbits[0]
                abit = m.abits[0]
                sum_bitw += size_product * wbit

                cycles_analog, cycles_digital = 0, 0
                for idx, wb in enumerate(m.wbits):
                    if len(m.wbits) > 1:
                        ch_out = m.mix_weight.alpha_weight[idx].sum()
                    else:
                        ch_out = torch.tensor(m.ch_out)
                    # Define dict whit shape infos used to model accelerators perf
                    conv_shape = {
                        'ch_in': m.ch_in,
                        'ch_out': ch_out,
                        'groups': m.groups,
                        'k_x': m.k_x,
                        'k_y': m.k_y,
                        'out_x': m.out_x,
                        'out_y': m.out_y,
                        }
                    if wb == 2:
                        cycles_analog = self.hw_model('analog', **conv_shape)
                    else:
                        cycles_digital = self.hw_model('digital', **conv_shape)
                if m.groups == 1:
                    cycles = max(cycles_analog, cycles_digital)
                else:
                    cycles = cycles_digital

                bita = memory_size * abit
                bitw = m.param_size * wbit
                sum_cycles += cycles
                sum_bita += bita
                sum_bitw += bitw
                layer_idx += 1
        return sum_cycles, sum_bita, sum_bitw


def quantmobilenetv1_fp(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 30, [[8]] * 30
    model = MobileNetV1(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', **kwargs)
    return model


def quantmobilenetv1_fp_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[8]] * 30, [[8]] * 30
    model = MobileNetV1(qm.FpConv2d, hw.diana(analog_speedup=5.),
                        archws, archas, qtz_fc='multi', **kwargs)
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    model.load_state_dict(fp_state_dict)

    # Fold bn
    model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(model)
    folded_model.train()  # Put folded model in train mode

    return folded_model


def quantmobilenetv1_w8a7_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 30, [[8]] * 30
    fp_model = MobileNetV1(qm.FpConv2d, hw.diana(analog_speedup=5.),
                           archws, archas, qtz_fc='multi', **kwargs)
    q_model = MobileNetV1(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=5.),
                          archws, archas, qtz_fc='multi', **kwargs)
    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantmobilenetv1_w2a7_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 30, [[2]] * 30
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    fp_model = MobileNetV1(qm.FpConv2d, hw.diana(analog_speedup=5.),
                           archws, archas, qtz_fc='multi', **kwargs)
    q_model = MobileNetV1(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=5.),
                          archws, archas, qtz_fc='multi', **kwargs)
    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantmobilenetv1_w2a7_true_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        print(f"The file {arch_cfg_path} does not exist.")
        raise FileNotFoundError

    archas, archws = [[7]] * 30, [[2]] * 30
    fp_model = MobileNetV1(qm.FpConv2d, hw.diana(analog_speedup=5.),
                           archws, archas, qtz_fc='multi', **kwargs)
    q_model = MobileNetV1(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=5.),
                          archws, archas, qtz_fc='multi', **kwargs)
    # Load pretrained fp state_dict
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    fp_model.load_state_dict(fp_state_dict)
    # Fold bn
    fp_model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(fp_model)
    folded_state_dict = folded_model.state_dict()

    # Delete fp and folded model
    del fp_model, folded_model

    # Translate folded fp state dict in a format compatible with quantized layers
    q_state_dict = utils.fpfold_to_q(folded_state_dict)
    # Load folded fp state dict in quantized model
    q_model.load_state_dict(q_state_dict, strict=False)

    # Init scale param
    utils.init_scale_param(q_model)

    return q_model


def quantmobilenetv1_minlat_foldbn(arch_cfg_path, **kwargs):
    ...


def quantmobilenetv1_minlat_max8_foldbn(arch_cfg_path, **kwargs):
    ...