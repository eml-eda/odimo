import copy
import math
from pathlib import Path

import torch
import torch.nn as nn

from . import utils
from . import quant_module as qm
from . import hw_models as hw

# MR
__all__ = [
    'quantres8_fp', 'quantres8_fp_foldbn',
    'quantres8_w8a8', 'quantres8_w8a8_nobn',
    'quantres20_w8a8',
    'quantres8_w5a8',
    'quantres8_w2a8', 'quantres8_w2a8_nobn',
    'quantres20_w2a8',
    'quantres8_w2a8_true', 'quantres8_w2a8_true_nobn',
    'quantres8_diana',
    'quantres8_diana5', 'quantres8_diana10', 'quantres8_diana100',
]


# MR
class Backbone20(nn.Module):

    def __init__(self, conv_func, input_size, bn, abits, wbits, **kwargs):
        super().__init__()
        self.bb_1_0 = BasicBlock(conv_func, 16, 16, wbits[:2], abits[:2], stride=1,
                                 bn=bn, **kwargs)
        self.bb_1_1 = BasicBlock(conv_func, 16, 16, wbits[2:4], abits[2:4], stride=1,
                                 bn=bn, **kwargs)
        self.bb_1_2 = BasicBlock(conv_func, 16, 16, wbits[4:6], abits[4:6], stride=1,
                                 bn=bn, **kwargs)
        self.bb_2_0 = BasicBlock(conv_func, 16, 32, wbits[6:9], abits[6:9], stride=2,
                                 bn=bn, **kwargs)
        self.bb_2_1 = BasicBlock(conv_func, 32, 32, wbits[9:11], abits[9:11], stride=1,
                                 bn=bn, **kwargs)
        self.bb_2_2 = BasicBlock(conv_func, 32, 32, wbits[11:13], abits[11:13], stride=1,
                                 bn=bn, **kwargs)
        self.bb_3_0 = BasicBlock(conv_func, 32, 64, wbits[13:16], abits[13:16], stride=2,
                                 bn=bn, **kwargs)
        self.bb_3_1 = BasicBlock(conv_func, 64, 64, wbits[16:18], abits[16:18], stride=1,
                                 bn=bn, **kwargs)
        self.bb_3_2 = BasicBlock(conv_func, 64, 64, wbits[18:20], abits[18:20], stride=1,
                                 bn=bn, **kwargs)
        self.pool = nn.AvgPool2d(kernel_size=8)

    def forward(self, x):
        x = self.bb_1_0(x)
        x = self.bb_1_1(x)
        x = self.bb_1_2(x)
        x = self.bb_2_0(x)
        x = self.bb_2_1(x)
        x = self.bb_2_2(x)
        x = self.bb_3_0(x)
        x = self.bb_3_1(x)
        x = self.bb_3_2(x)
        x = self.pool(x)
        return x


# MR
class BackboneTiny(nn.Module):

    def __init__(self, conv_func, input_size, bn, abits, wbits, **kwargs):
        super().__init__()
        self.bb_1 = BasicBlock(conv_func, 16, 16, wbits[:2], abits[:2], stride=1,
                               bn=bn, **kwargs)
        self.bb_2 = BasicBlock(conv_func, 16, 32, wbits[2:5], abits[2:5], stride=2,
                               bn=bn, **kwargs)
        self.bb_3 = BasicBlock(conv_func, 32, 64, wbits[5:7], abits[5:7], stride=2,
                               bn=bn, **kwargs)
        self.pool = nn.AvgPool2d(kernel_size=8)

    def forward(self, x):
        x = self.bb_1(x)
        x = self.bb_2(x)
        x = self.bb_3(x)
        x = self.pool(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, conv_func, inplanes, planes, archws, archas, stride=1,
                 downsample=None, bn=True, **kwargs):
        self.bn = bn
        super().__init__()
        self.conv1 = conv_func(inplanes, planes, archws[0], archas[0], kernel_size=3, stride=stride,
                               groups=1, padding=1, bias=False, **kwargs)
        if bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_func(planes, planes, archws[1], archas[1], kernel_size=3, stride=1,
                               groups=1, padding=1, bias=False, **kwargs)
        if bn:
            self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or inplanes != planes:
            self.downsample = conv_func(
                inplanes, planes, archws[-1], archas[-1], kernel_size=1,
                groups=1, stride=stride, bias=False, **kwargs)
            if bn:
                self.bn_ds = nn.BatchNorm2d(planes)
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            residual = x
        else:
            residual = x

        out = self.conv1(x)
        if self.bn:
            out = self.bn1(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
            if self.bn:
                residual = self.bn_ds(residual)

        out += residual

        return out


class ResNet20(nn.Module):

    def __init__(self, conv_func, hw_model, archws, archas, qtz_fc=None,
                 input_size=32, num_classes=10, bn=True, **kwargs):
        print('archas: {}'.format(archas))
        print('archws: {}'.format(archws))

        self.inplanes = 16
        self.conv_func = conv_func
        self.hw_model = hw_model
        self.search_types = ['fixed', 'mixed', 'multi']
        if qtz_fc in self.search_types:
            self.qtz_fc = qtz_fc
        else:
            self.qtz_fc = False
        self.bn = bn
        super().__init__()

        # Model
        self.conv1 = conv_func(3, 16, abits=archas[0], wbits=archws[0],
                               kernel_size=3, stride=1, bias=False, padding=1,
                               groups=1, first_layer=False, **kwargs)
        if bn:
            self.bn1 = nn.BatchNorm2d(16)
        self.backbone = Backbone20(
            conv_func, input_size, bn, abits=archas[1:-1], wbits=archws[1:-1], **kwargs)
        self.fc = conv_func(
            64, num_classes, abits=archas[-1], wbits=archws[-1],
            kernel_size=1, stride=1, groups=1, bias=True, fc=self.qtz_fc, **kwargs)

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

    def forward(self, x):
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)

        x = self.backbone(x)

        x = x if self.qtz_fc else x.view(x.size(0), -1)
        x = self.fc(x)[:, :, 0, 0] if self.qtz_fc else self.fc(x)

        return x


class TinyMLResNet(nn.Module):

    def __init__(self, conv_func, hw_model, archws, archas, qtz_fc=None,
                 input_size=32, num_classes=10, bn=True, **kwargs):
        print('archas: {}'.format(archas))
        print('archws: {}'.format(archws))

        self.inplanes = 16
        self.conv_func = conv_func
        self.hw_model = hw_model
        self.search_types = ['fixed', 'mixed', 'multi']
        if qtz_fc in self.search_types:
            self.qtz_fc = qtz_fc
        else:
            self.qtz_fc = False
        self.bn = bn
        super().__init__()

        # Model
        self.conv1 = conv_func(3, 16, abits=archas[0], wbits=archws[0],
                               kernel_size=3, stride=1, bias=False, padding=1,
                               groups=1, first_layer=False, **kwargs)
        if bn:
            self.bn1 = nn.BatchNorm2d(16)
        self.backbone = BackboneTiny(
            conv_func, input_size, bn, abits=archas[1:-1], wbits=archws[1:-1], **kwargs)
        self.fc = conv_func(
            64, num_classes, abits=archas[-1], wbits=archws[-1],
            kernel_size=1, stride=1, groups=1, bias=True, fc=self.qtz_fc, **kwargs)

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

    def forward(self, x):
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)

        x = self.backbone(x)

        x = x if self.qtz_fc else x.view(x.size(0), -1)
        x = self.fc(x)[:, :, 0, 0] if self.qtz_fc else self.fc(x)

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

                if wbit == 2:
                    cycles = size_product / self.hw_model('analog')
                else:
                    cycles = size_product / self.hw_model('digital')

                bita = memory_size * abit
                bitw = m.param_size * wbit
                sum_cycles += cycles
                sum_bita += bita
                sum_bitw += bitw
                layer_idx += 1
        return sum_cycles, sum_bita, sum_bitw


def _load_arch(arch_path, names_nbits):
    checkpoint = torch.load(arch_path)
    state_dict = checkpoint['state_dict']
    best_arch, worst_arch = {}, {}
    for name in names_nbits.keys():
        best_arch[name], worst_arch[name] = [], []
    for name, params in state_dict.items():
        name = name.split('.')[-1]
        if name in names_nbits.keys():
            alpha = params.cpu().numpy()
            assert names_nbits[name] == alpha.shape[0]
            best_arch[name].append(alpha.argmax())
            worst_arch[name].append(alpha.argmin())

    return best_arch, worst_arch


# MR
def _load_arch_multi_prec(arch_path):
    checkpoint = torch.load(arch_path)
    state_dict = checkpoint['state_dict']
    best_arch, worst_arch = {}, {}
    best_arch['alpha_activ'], worst_arch['alpha_activ'] = [], []
    best_arch['alpha_weight'], worst_arch['alpha_weight'] = [], []
    for name, params in state_dict.items():
        name = name.split('.')[-1]
        if name == 'alpha_activ':
            alpha = params.cpu().numpy()
            best_arch[name].append(alpha.argmax())
            worst_arch[name].append(alpha.argmin())
        elif name == 'alpha_weight':
            alpha = params.cpu().numpy()
            best_arch[name].append(alpha.argmax(axis=0))
            worst_arch[name].append(alpha.argmin(axis=0))

    return best_arch, worst_arch


# MR
def _load_weights(arch_path):
    checkpoint = torch.load(arch_path)
    state_dict = checkpoint['state_dict']
    weights = {}
    for name, params in state_dict.items():
        type_ = name.split('.')[-1]
        if type_ == 'weight':
            weight = params.cpu().numpy()
            weights[name] = weight
        elif name == 'bias':
            bias = params.cpu().numpy()
            weights[name] = bias

    return weights


# MR
def _load_alpha_state_dict(arch_path):
    checkpoint = torch.load(arch_path)
    state_dict = checkpoint['state_dict']
    alpha_state_dict = dict()
    for name, params in state_dict.items():
        full_name = name
        name = name.split('.')[-1]
        if name == 'alpha_activ' or name == 'alpha_weight':
            alpha_state_dict[full_name] = params

    return alpha_state_dict


# MR
def _load_alpha_state_dict_as_mp(arch_path, model):
    checkpoint = torch.load(arch_path)
    state_dict = checkpoint['state_dict']
    alpha_state_dict = dict()
    for name, params in state_dict.items():
        full_name = name
        name = name.split('.')[-1]
        if name == 'alpha_activ':
            alpha_state_dict[full_name] = params
        elif name == 'alpha_weight':
            mp_params = torch.tensor(model.state_dict()[full_name])
            mp_params[0] = params[0]
            mp_params[1] = params[1]
            mp_params[2] = params[2]
            alpha_state_dict[full_name] = mp_params

    return alpha_state_dict


# MR
def _remove_alpha(state_dict):
    weight_state_dict = copy.deepcopy(state_dict)
    for name, params in state_dict.items():
        full_name = name
        name = name.split('.')[-1]
        if name == 'alpha_activ':
            weight_state_dict.pop(full_name)
        elif name == 'alpha_weight':
            weight_state_dict.pop(full_name)

    return weight_state_dict


def quantres8_fp(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 10, [[8]] * 10
    model = TinyMLResNet(qm.FpConv2d, hw.diana(analog_speedup=5.),
                         archws, archas, qtz_fc='multi', **kwargs)
    return model


def quantres8_fp_foldbn(arch_cfg_path, **kwargs):
    # Check `arch_cfg_path` existence
    if not Path(arch_cfg_path).exists():
        raise FileNotFoundError

    archas, archws = [[8]] * 10, [[8]] * 10
    model = TinyMLResNet(qm.FpConv2d, hw.diana(analog_speedup=5.),
                         archws, archas, qtz_fc='multi', **kwargs)
    fp_state_dict = torch.load(arch_cfg_path)['state_dict']
    model.load_state_dict(fp_state_dict)

    model.eval()  # Model must be in eval mode to fold bn
    folded_model = utils.fold_bn(model)
    folded_model.train()  # Put back the model in train mode

    return folded_model


def quantres8_w8a8(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 10, [[8]] * 10
    s_up = kwargs.pop('analog_speedup', 5.)
    model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                         archws, archas, qtz_fc='multi', **kwargs)
    return model


def quantres20_w8a8(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 22, [[8]] * 22
    s_up = kwargs.pop('analog_speedup', 5.)
    model = ResNet20(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                     archws, archas, qtz_fc='multi', **kwargs)
    return model


def quantres8_w8a8_nobn(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 10, [[8]] * 10
    s_up = kwargs.pop('analog_speedup', 5.)
    model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                         archws, archas, qtz_fc='multi', bn=False, **kwargs)
    return model


def quantres8_w5a8(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 10, [[5]] * 10
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)

    # Build Model
    model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                         archws, archas, qtz_fc='multi', **kwargs)
    state_dict = torch.load(arch_cfg_path)['state_dict']
    model.load_state_dict(state_dict)
    return model


def quantres8_w2a8(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 10, [[2]] * 10
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)

    # Build Model
    model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                         archws, archas, qtz_fc='multi', **kwargs)
    # state_dict = torch.load(arch_cfg_path)['state_dict']
    # model.load_state_dict(state_dict)
    return model


def quantres20_w2a8(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 22, [[2]] * 22
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)

    # Build Model
    model = ResNet20(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                     archws, archas, qtz_fc='multi', **kwargs)
    # state_dict = torch.load(arch_cfg_path)['state_dict']
    # model.load_state_dict(state_dict)
    return model


def quantres8_w2a8_nobn(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 10, [[2]] * 10
    # Set first and last layer weights precision to 8bit
    archws[0] = [8]
    archws[-1] = [8]
    s_up = kwargs.pop('analog_speedup', 5.)

    # Build Model
    model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                         archws, archas, qtz_fc='multi', bn=False, **kwargs)
    # state_dict = torch.load(arch_cfg_path)['state_dict']
    # model.load_state_dict(state_dict)
    return model


def quantres8_w2a8_true(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 10, [[2]] * 10
    s_up = kwargs.pop('analog_speedup', 5.)

    # Build Model
    model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                         archws, archas, qtz_fc='multi', **kwargs)
    # state_dict = torch.load(arch_cfg_path)['state_dict']
    # model.load_state_dict(state_dict)
    return model


def quantres8_w2a8_true_nobn(arch_cfg_path, **kwargs):
    archas, archws = [[8]] * 10, [[2]] * 10
    s_up = kwargs.pop('analog_speedup', 5.)

    # Build Model
    model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=s_up),
                         archws, archas, qtz_fc='multi', bn=False, **kwargs)
    # state_dict = torch.load(arch_cfg_path)['state_dict']
    # model.load_state_dict(state_dict)
    return model


def quantres8_diana5(arch_cfg_path, **kwargs):
    return quantres8_diana(arch_cfg_path, **kwargs)


def quantres8_diana10(arch_cfg_path, **kwargs):
    return quantres8_diana(arch_cfg_path, **kwargs)


def quantres8_diana100(arch_cfg_path, **kwargs):
    return quantres8_diana(arch_cfg_path, **kwargs)


# ToDO
# qtz_fc: None or 'fixed' or 'mixed' or 'multi'
def quantres8_diana(arch_cfg_path, **kwargs):
    wbits, abits = [2, 8], [8]

    # ## This block of code is only necessary to comply with the underlying EdMIPS code ##
    best_arch, worst_arch = _load_arch_multi_prec(arch_cfg_path)
    archas = [abits for a in best_arch['alpha_activ']]
    archws = [wbits for w_ch in best_arch['alpha_weight']]
    if len(archws) == 9:
        # Case of fixed-precision on last fc layer
        archws.append(8)
    assert len(archas) == 10  # 10 insead of 8 because conv1 and fc activations are also quantized
    assert len(archws) == 10  # 10 instead of 8 because conv1 and fc weights are also quantized
    ##

    model = TinyMLResNet(qm.QuantMultiPrecActivConv2d, hw.diana(analog_speedup=5.),
                         archws, archas, qtz_fc='multi', **kwargs)

    if kwargs['fine_tune']:
        # Load all weights
        state_dict = torch.load(arch_cfg_path)['state_dict']
        model.load_state_dict(state_dict)
    else:
        # Load only alphas weights
        alpha_state_dict = _load_alpha_state_dict(arch_cfg_path)
        model.load_state_dict(alpha_state_dict, strict=False)
    return model
