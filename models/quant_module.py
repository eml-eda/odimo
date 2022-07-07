from __future__ import print_function

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.parameter import Parameter


# DJP (TODO: test symmetric quant)
class _channel_sym_min_max_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, bit):
        ch_max, _ = x.view(x.size(0), -1).abs().max(1)
        return _channel_min_max_quantize_common(x, -ch_max, ch_max, bit)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


# MR:
class _bias_asym_min_max_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, bit):
        bias_max = x.max()
        bias_min = x.min()
        return _bias_min_max_quantize_common(x, bias_min, bias_max, bit)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


# MR:
class _bias_sym_min_max_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, bit):
        bias_max = x.abs().max()
        return _bias_min_max_quantize_common(x, -bias_max, bias_max, bit)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


# MR
def _bias_min_max_quantize_common(x, ch_min, ch_max, bit):
    bias_range = ch_max - ch_min
    n_steps = 2 ** bit - 1
    S_bias = bias_range / n_steps
    y = (x / S_bias).round() * S_bias

    return y


# DJP: (Check for DW)
class _channel_asym_min_max_quantize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, bit):
        ch_max, _ = x.view(x.size(0), -1).max(1)
        ch_min, _ = x.view(x.size(0), -1).min(1)
        return _channel_min_max_quantize_common(x, ch_min, ch_max, bit)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


# DJP (TODO: are clones necessary?)
def _channel_min_max_quantize_common(x, ch_min, ch_max, bit):
    # ## old version
    # ch_max_mat = ch_max.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(x.size())
    # ch_min_mat = ch_min.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(x.size())
    # y = x.clone()
    # clamp (this shouldn't do anything right?)
    # y = torch.max(y, ch_min_mat)
    # y = torch.min(y, ch_max_mat)
    # scale
    # range_mat = ch_max_mat - ch_min_mat
    # range_mat.masked_fill_(range_mat.eq(0), 1)
    # n_steps = 2 ** bit - 1
    # S_w = range_mat / n_steps
    # y = y.div(S_w).round().clone()
    # y = y.mul(S_w)

    # ## new version
    if bit != 0:
        ch_range = ch_max - ch_min
        ch_range.masked_fill_(ch_range.eq(0), 1)
        n_steps = 2 ** bit - 1
        S_w = ch_range / n_steps
        S_w = S_w.view((x.size(0), 1, 1, 1))
        y = x.div(S_w).round().mul(S_w)
    else:
        y = torch.zeros(x.shape, device=x.device)

    return y


# MR
class _prune_channels(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sw):
        ctx.save_for_backward(sw)
        sw_bin = (torch.argmax(sw, dim=0) > 0).float()
        all_zero = torch.count_nonzero(sw_bin) == 0
        return sw_bin + all_zero.float()

    @staticmethod
    def backward(ctx, grad_output):
        sw, = ctx.saved_tensors
        # Adapt grad_output to the shape of sw
        return grad_output.expand_as(sw)


# DJP
def asymmetric_linear_quantization_scale_factor(num_bits, saturation_min, saturation_max):
    n = 2 ** num_bits - 1
    return n / (saturation_max - saturation_min)


# DJP
def linear_quantize(input, scale_factor, inplace=False):
    if inplace:
        input.mul_(scale_factor).floor_()
        return input
    return torch.floor(scale_factor * input)


# DJP
def linear_dequantize(input, scale_factor, inplace=False):
    if inplace:
        input.div_(scale_factor)
        return input
    return input / scale_factor


# DJP
def clamp(x, min, max, inplace=False):
    if inplace:
        x.clamp_(min, max)
        return x
    return torch.clamp(x, min, max)


@torch.fx.wrap
def memory_size(in_shape):
    return torch.tensor(in_shape[1] * in_shape[2] * in_shape[3] * 1e-3, dtype=torch.float)


@torch.fx.wrap
def size_product(filter_size, in_shape):
    return torch.tensor(filter_size * in_shape[-1] * in_shape[-2], dtype=torch.float)


# DJP
class LearnedClippedLinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, dequantize, inplace):
        ctx.save_for_backward(input, clip_val)
        if inplace:
            ctx.mark_dirty(input)
        scale_factor = asymmetric_linear_quantization_scale_factor(num_bits, 0, clip_val.data[0])
        output = clamp(input, 0, clip_val.data[0], inplace)
        output = linear_quantize(output, scale_factor, inplace)
        if dequantize:
            output = linear_dequantize(output, scale_factor, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.masked_fill_(input.le(0), 0)
        grad_input.masked_fill_(input.ge(clip_val.data[0]), 0)

        grad_alpha = grad_output.clone()
        grad_alpha.masked_fill_(input.lt(clip_val.data[0]), 0)
#        grad_alpha[input.lt(clip_val.data[0])] = 0
        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        # Straight-through estimator for the scale factor calculation
        return grad_input, grad_alpha, None, None, None


# DJP (w.r.t Manuele's code I changed inplace to false to avoid error)
class LearnedClippedLinearQuantization(nn.Module):
    def __init__(self, num_bits, init_act_clip_val=6, dequantize=True, inplace=False):
        super(LearnedClippedLinearQuantization, self).__init__()
        self.num_bits = num_bits
        self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]))
        self.dequantize = dequantize
        self.inplace = inplace

    def forward(self, input):
        input = LearnedClippedLinearQuantizeSTE.apply(
            input, self.clip_val, self.num_bits, self.dequantize, self.inplace)
        return input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(num_bits={1}, clip_val={2}{3})'.format(
            self.__class__.__name__, self.num_bits, self.clip_val, inplace_str)


# DJP
class FQQuantizationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits, inplace, lower_bound):
        if inplace:
            ctx.mark_dirty(x)
        # Number of positive quantization levels
        n = 2**(num_bits-1) - 1
        # Hardtanh
        output = clamp(x, lower_bound, 1, inplace)
        # Multiply by number of levels
        output = output * n
        # Round
        output = torch.round(output)
        # Divide by number of levels
        output = output / n

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # x, scale_param = ctx.saved_tensors
        grad_input = grad_output.clone()
        # grad_input.masked_fill_(x.le(0), 0)
        # grad_input.masked_fill_(x.ge(scale_param.data[0]), 0)

        # grad_alpha = grad_output.clone()
        # grad_alpha.masked_fill_(x.lt(scale_param.data[0]), 0)
        # grad_alpha[input.lt(clip_val.data[0])] = 0
        # grad_alpha = grad_alpha.sum().expand_as(scale_param)

        return grad_input, None, None, None


# MR (Ref: https://arxiv.org/abs/1912.09356)
class FQActQuantization(nn.Module):
    def __init__(self, num_bits, init_scale_param=0.,
                 train_scale_param=False, inplace=False):
        super().__init__()
        self.num_bits = num_bits
        self.train_scale_param = train_scale_param
        self.n_s = 1  # One scale-param per layer
        self.scale_param = nn.Parameter(torch.Tensor(self.n_s), requires_grad=train_scale_param)
        # To be compliant with LearnedClippedLinearQuantization where init_clip_val=6
        # we set here scale_param to be ln(6) (6/e^s = 1)
        init_scale_param = math.log(6)
        self.scale_param.data.fill_(init_scale_param)
        self.inplace = inplace

    def forward(self, x):
        # Having a positive scale factor is preferable to avoid instabilities
        exp_scale_param = torch.exp(self.scale_param)
        x_scaled = x / exp_scale_param.view(self.n_s, 1, 1, 1)
        # Quantize
        x_q = FQQuantizationSTE.apply(x_scaled, self.num_bits+1, self.inplace, 0)
        # Multiply by scale factor
        x_deq = x_q * exp_scale_param.view(self.n_s, 1, 1, 1)
        return x_deq

    def __repr__(self):
        return f'{self.__class__.__name__}(num_bits={self.num_bits}, \
            scale_param={self.scale_param})'


# MR (Ref: https://arxiv.org/abs/1912.09356)
class FQConvWeightQuantization(nn.Module):
    def __init__(self, cout, k_size, num_bits, init_scale_param=0.,
                 train_scale_param=False, inplace=False):
        super().__init__()
        self.cout = cout
        self.k_size = k_size
        self.num_bits = num_bits
        self.train_scale_param = train_scale_param
        # self.n_s = cout  # One scale-param per channel
        self.n_s = 1  # One scale-param per layer
        self.scale_param = nn.Parameter(torch.Tensor(self.n_s), requires_grad=train_scale_param)
        # Choose s in such a way the [-1, 0, 1] values are equiprobable
        mu = torch.tensor([0.])
        std = math.sqrt(2/torch.tensor([self.cout * self.k_size]))
        n = Normal(mu, std)  # mean, std
        # P[x / e^s < -1/2] = p -> P[x < -1/2 * e^s] = p
        # icdf(p) = -1/2 * e^s -> s = ln(-2 * icdf(p))
        p = torch.tensor([1/(2**num_bits-1)])
        init_scale_param = math.log(-2 * n.icdf(p))
        self.scale_param.data.fill_(init_scale_param)
        self.inplace = inplace

    def forward(self, x):
        # Having a positive scale factor is preferable to avoid instabilities
        exp_scale_param = torch.exp(self.scale_param)
        x_scaled = x / exp_scale_param.view(self.n_s, 1, 1, 1)
        # Quantize
        x_q = FQQuantizationSTE.apply(x_scaled, self.num_bits, self.inplace, -1)
        # Multiply by scale factor
        x_deq = x_q * exp_scale_param.view(self.n_s, 1, 1, 1)
        return x_deq

    def __repr__(self):
        return f'{self.__class__.__name__}(num_bits={self.num_bits}, \
            scale_param={self.scale_param})'


# MR
class QuantMultiPrecActivConv2d(nn.Module):

    def __init__(self, inplane, outplane, wbits=None, abits=None, fc=None, **kwargs):
        super().__init__()
        self.fine_tune = kwargs.pop('fine_tune', False)
        self.first_layer = kwargs.pop('first_layer', False)
        self.fc = fc

        self.abits = abits
        self.wbits = wbits

        self.search_types = ['fixed', 'mixed', 'multi']
        if fc in self.search_types:
            self.fc = fc
        else:
            self.fc = False
        self.mix_activ = QuantPaCTActiv(abits)
        # self.mix_activ = QuantFQActiv(abits)
        if not fc:
            self.mix_weight = QuantMultiPrecConv2d(inplane, outplane, wbits, **kwargs)
        else:
            # For the final fc layer the pruning bit-width (i.e., 0) makes no sense
            _wbits = copy.deepcopy(wbits)
            if 0 in _wbits:
                _wbits.remove(0)
            # If the layer is fc we can use:
            if self.fc == 'fixed':
                # - Fixed quantization on 8bits
                self.mix_weight = QuantMixChanConv2d(inplane, outplane, 8, **kwargs)
            elif self.fc == 'mixed':
                # - Mixed-precision search
                self.mix_weight = QuantMixChanConv2d(inplane, outplane, _wbits, **kwargs)
            elif self.fc == 'multi':
                # - Multi-precision search
                self.mix_weight = QuantMultiPrecConv2d(
                    inplane, outplane, _wbits, train_scale_param=True, **kwargs)
            else:
                raise ValueError(f"Unknown fc search, possible values are {self.search_types}")

        # complexities
        stride = kwargs['stride'] if 'stride' in kwargs else 1
        if isinstance(kwargs['kernel_size'], tuple):
            kernel_size = kwargs['kernel_size'][0] * kwargs['kernel_size'][1]
        else:
            kernel_size = kwargs['kernel_size'] * kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size / kwargs['groups'] * 1e-6
        self.filter_size = self.param_size / float(stride ** 2.0)
        self.register_buffer('size_product', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        in_shape = input.shape
        # tmp = torch.tensor(in_shape[1] * in_shape[2] * in_shape[3] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(memory_size(in_shape))
        # tmp = torch.tensor(self.filter_size * in_shape[-1] * in_shape[-2], dtype=torch.float)
        self.size_product.copy_(size_product(self.filter_size, in_shape))
        if not self.first_layer:
            out = self.mix_activ(input)
        else:
            out = _channel_asym_min_max_quantize.apply(input, 8)
        out = self.mix_weight(out)
        return out


# MR
class QuantFQActiv(nn.Module):

    def __init__(self, bits):
        super().__init__()
        if type(bits) == int:
            self.bits = [bits]
        else:
            self.bits = bits
        self.alpha_activ = Parameter(torch.Tensor(len(self.bits)), requires_grad=False)
        self.alpha_activ.data.fill_(0.01)
        self.mix_activ = nn.ModuleList()
        for bit in self.bits:
            self.mix_activ.append(FQActQuantization(num_bits=bit, train_scale_param=True))

    def forward(self, input):
        outs = []
        # self.alpha_activ = torch.nn.Parameter(clamp(self.alpha_activ,-100,+100))
        sw = F.one_hot(torch.argmax(self.alpha_activ), num_classes=len(self.bits))
        for i, branch in enumerate(self.mix_activ):
            # torch.nan_to_num() necessary to avoid nan in the output when multiplying by zero
            outs.append(torch.nan_to_num(branch(input)) * sw[i])
        activ = sum(outs)
        return activ


# MR
class QuantPaCTActiv(nn.Module):

    def __init__(self, bits):
        super(QuantPaCTActiv, self).__init__()
        if type(bits) == int:
            self.bits = [bits]
        else:
            self.bits = bits
        self.alpha_activ = Parameter(torch.Tensor(len(self.bits)), requires_grad=False)
        self.alpha_activ.data.fill_(0.01)
        self.mix_activ = nn.ModuleList()
        for bit in self.bits:
            self.mix_activ.append(LearnedClippedLinearQuantization(num_bits=bit))

    def forward(self, input):
        outs = []
        # self.alpha_activ = torch.nn.Parameter(clamp(self.alpha_activ,-100,+100))
        sw = F.one_hot(torch.argmax(self.alpha_activ), num_classes=len(self.bits))
        for i, branch in enumerate(self.mix_activ):
            # torch.nan_to_num() necessary to avoid nan in the output when multiplying by zero
            outs.append(torch.nan_to_num(branch(input)) * sw[i])
        activ = sum(outs)
        return activ


# MR
class QuantMultiPrecConv2d(nn.Module):

    def __init__(self, inplane, outplane, bits, **kwargs):
        super().__init__()
        kwargs.pop('abit', None)
        if type(bits) == int:
            self.bits = [bits]
        else:
            self.bits = bits
        self.cout = outplane
        self.alpha_weight = Parameter(torch.Tensor(len(self.bits), self.cout), requires_grad=False)
        self.alpha_weight.data.fill_(0.01)

        if isinstance(kwargs['kernel_size'], tuple):
            k_size = kwargs['kernel_size'][0] * kwargs['kernel_size'][1]
        else:
            k_size = kwargs['kernel_size'] * kwargs['kernel_size']

        # Quantizer
        self.mix_weight = nn.ModuleList()
        self.train_scale_param = kwargs.pop('train_scale_param', True)
        for bit in self.bits:
            self.mix_weight.append(
                FQConvWeightQuantization(
                    outplane, k_size,
                    num_bits=bit,
                    train_scale_param=self.train_scale_param))

        self.conv = nn.Conv2d(inplane, outplane, **kwargs)

    def forward(self, input):
        mix_quant_weight = []
        sw = F.one_hot(torch.argmax(self.alpha_weight, dim=0), num_classes=len(self.bits)).t()
        conv = self.conv
        weight = conv.weight
        bias = getattr(conv, 'bias', None)
        for i, bit in enumerate(self.bits):
            quant_weight = self.mix_weight[i](weight)
            scaled_quant_weight = quant_weight * sw[i].view((self.cout, 1, 1, 1))
            mix_quant_weight.append(scaled_quant_weight)
        mix_quant_weight = sum(mix_quant_weight)
        if bias is not None:
            quant_bias = _bias_sym_min_max_quantize.apply(bias, 32)
        else:
            quant_bias = bias
        out = F.conv2d(
            input, mix_quant_weight, quant_bias, conv.stride,
            conv.padding, conv.dilation, conv.groups)
        return out


# MR
class QuantMixActivChanConv2d(nn.Module):

    def __init__(self, inplane, outplane, wbits, abits, fc=False, **kwargs):
        super(QuantMixActivChanConv2d, self).__init__()
        self.abits = abits
        self.wbit = wbits
        self.fc = False

        self.first_layer = kwargs.pop('first_layer', False)

        self.mix_activ = QuantPaCTActiv(abits)
        if not self.fc:
            self.mix_weight = QuantMixChanConv2d(inplane, outplane, bits=wbits, **kwargs)
        else:
            # If the layer is fc we use fixed quantization on 8bits
            self.mix_weight = QuantMixChanConv2d(inplane, outplane, 8, **kwargs)
        # complexities
        stride = kwargs['stride'] if 'stride' in kwargs else 1
        if isinstance(kwargs['kernel_size'], tuple):
            kernel_size = kwargs['kernel_size'][0] * kwargs['kernel_size'][1]
        else:
            kernel_size = kwargs['kernel_size'] * kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size / kwargs['groups'] * 1e-6
        self.filter_size = self.param_size / float(stride ** 2.0)
        self.register_buffer('size_product', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        in_shape = input.shape
        tmp = torch.tensor(in_shape[1] * in_shape[2] * in_shape[3] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(self.filter_size * in_shape[-1] * in_shape[-2], dtype=torch.float)
        self.size_product.copy_(tmp)
        if not self.first_layer:
            out = self.mix_activ(input)
            out = self.mix_weight(out)
        else:
            out = _channel_asym_min_max_quantize.apply(input, 8)
            out = self.mix_weight(out)
        return out


# MR
class FpConv2d(nn.Module):

    def __init__(self, inplane, outplane, wbits, abits, first_layer=False, **kwargs):
        super(FpConv2d, self).__init__()
        self.abit = abits
        self.wbits = wbits

        self.first_layer = first_layer

        self.fine_tune = kwargs.pop('fine_tune', False)
        self.fc = kwargs.pop('fc', False)

        self.conv = nn.Conv2d(inplane, outplane, **kwargs)
        self.relu = nn.ReLU()
        # complexities
        stride = kwargs['stride'] if 'stride' in kwargs else 1
        if isinstance(kwargs['kernel_size'], tuple):
            kernel_size = kwargs['kernel_size'][0] * kwargs['kernel_size'][1]
        else:
            kernel_size = kwargs['kernel_size'] * kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size * 1e-6
        self.filter_size = self.param_size / float(stride ** 2.0)
        self.register_buffer('size_product', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input):
        in_shape = input.shape
        # tmp = torch.tensor(in_shape[1] * in_shape[2] * in_shape[3] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(memory_size(in_shape))
        # tmp = torch.tensor(self.filter_size * in_shape[-1] * in_shape[-2], dtype=torch.float)
        self.size_product.copy_(size_product(self.filter_size, in_shape))
        if not self.first_layer:
            out = self.conv(self.relu(input))
        else:
            out = self.conv(input)
        return out


# MR
class QuantMixChanConv2d(nn.Module):

    def __init__(self, inplane, outplane, bits, **kwargs):
        super(QuantMixChanConv2d, self).__init__()
        self.bits = bits
        self.outplane = outplane

        kwargs.pop('alpha_init', None)

        self.fine_tune = kwargs.pop('fine_tune', False)
        self.conv = nn.Conv2d(inplane, outplane, **kwargs)

    def forward(self, input):
        conv = self.conv
        bias = getattr(conv, 'bias', None)
        quant_weight = _channel_asym_min_max_quantize.apply(conv.weight, self.bits)
        if bias is not None:
            quant_bias = _bias_sym_min_max_quantize.apply(bias, 32)
        else:
            quant_bias = bias
        out = F.conv2d(
            input, quant_weight, quant_bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        return out


# DJP
class MixQuantPaCTActiv(nn.Module):

    def __init__(self, bits, gumbel=False):
        super().__init__()
        self.bits = bits
        self.gumbel = gumbel
        self.alpha_activ = Parameter(torch.Tensor(len(self.bits)))
        self.alpha_activ.data.fill_(0.01)
        self.mix_activ = nn.ModuleList()
        for bit in self.bits:
            self.mix_activ.append(LearnedClippedLinearQuantization(num_bits=bit))

    def forward(self, input, temp, is_hard):
        outs = []
        # self.alpha_activ = torch.nn.Parameter(clamp(self.alpha_activ,-100,+100))
        if not self.gumbel:
            sw = F.softmax(self.alpha_activ/temp, dim=0)
        else:
            # If is_hard is True the output is one-hot
            sw = F.gumbel_softmax(self.alpha_activ, tau=temp, hard=is_hard, dim=0)
        for i, branch in enumerate(self.mix_activ):
            outs.append(branch(input) * sw[i])
        activ = sum(outs)
        return activ


# DJP
class MixQuantChanConv2d(nn.Module):

    def __init__(self, inplane, outplane, bits, **kwargs):
        super(MixQuantChanConv2d, self).__init__()
        assert not kwargs['bias']
        self.bits = bits
        self.alpha_weight = Parameter(torch.Tensor(len(self.bits)))
        self.alpha_weight.data.fill_(0.01)
        self.conv_list = nn.ModuleList()
        for bit in self.bits:
            assert 0 < bit < 32
            self.conv_list.append(nn.Conv2d(inplane, outplane, **kwargs))

    def forward(self, input):
        mix_quant_weight = []
        sw = F.softmax(self.alpha_weight, dim=0)
        for i, bit in enumerate(self.bits):
            weight = self.conv_list[i].weight
            quant_weight = _channel_asym_min_max_quantize.apply(weight, bit)
            scaled_quant_weight = quant_weight * sw[i]
            mix_quant_weight.append(scaled_quant_weight)
        mix_quant_weight = sum(mix_quant_weight)
        conv = self.conv_list[0]
        out = F.conv2d(
            input, mix_quant_weight, conv.bias, conv.stride,
            conv.padding, conv.dilation, conv.groups)
        return out


# DJP
class SharedMixQuantChanConv2d(nn.Module):

    def __init__(self, inplane, outplane, bits, gumbel=False, **kwargs):
        super(SharedMixQuantChanConv2d, self).__init__()
        self.bits = bits
        self.gumbel = gumbel
        if isinstance(kwargs['kernel_size'], tuple):
            kernel_size = kwargs['kernel_size'][0] * kwargs['kernel_size'][1]
        else:
            kernel_size = kwargs['kernel_size'] * kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size / kwargs['groups'] * 1e-6
        self.alpha_weight = Parameter(torch.Tensor(len(self.bits)))
        self.alpha_init = kwargs.pop('alpha_init', 'same')
        if self.alpha_init == 'same':
            self.alpha_weight.data.fill_(0.01)
        elif self.alpha_init == 'scaled':
            max_prec = max(self.bits)
            for i in range(len(self.bits)):
                self.alpha_weight.data[i].fill_(self.bits[i] / max_prec)
        self.conv = nn.Conv2d(inplane, outplane, **kwargs)

    def forward(self, input, temp, is_hard):
        mix_quant_weight = []
        mix_wbit = 0
        # self.alpha_weight = torch.nn.Parameter(clamp(self.alpha_weight, -100, +100))
        if not self.gumbel:
            sw = F.softmax(self.alpha_weight / temp, dim=0)
        else:
            # If is_hard is True the output is one-hot
            sw = F.gumbel_softmax(self.alpha_weight, tau=temp, hard=is_hard, dim=0)
        conv = self.conv
        weight = conv.weight
        bias = getattr(conv, 'bias', None)
        for i, bit in enumerate(self.bits):
            quant_weight = _channel_asym_min_max_quantize.apply(weight, bit)
            scaled_quant_weight = quant_weight * sw[i]
            mix_quant_weight.append(scaled_quant_weight)
            # Complexity
            mix_wbit += sw[i] * bit
        if bias is not None:
            quant_bias = _bias_sym_min_max_quantize.apply(bias, 32)
        else:
            quant_bias = bias
        mix_quant_weight = sum(mix_quant_weight)
        out = F.conv2d(
            input, mix_quant_weight, quant_bias, conv.stride,
            conv.padding, conv.dilation, conv.groups)
        # Measure weight complexity for reg-loss
        w_complexity = mix_wbit * self.param_size
        return out, w_complexity


# DJP
class SharedMultiPrecConv2d(nn.Module):

    def __init__(self, inplane, outplane, bits, gumbel=False, **kwargs):
        super().__init__()
        self.bits = bits
        self.gumbel = gumbel
        self.cout = outplane

        self.alpha_weight = Parameter(torch.Tensor(len(self.bits), self.cout))
        # Alpha init
        self.alpha_init = kwargs.pop('alpha_init', 'same')
        if self.alpha_init == 'same' or self.alpha_init is None:
            if self.gumbel:
                val_equiprob = 1.0 / len(self.bits)
                init_logit = math.log(val_equiprob/(1-val_equiprob))
            else:
                init_logit = 0.01
            self.alpha_weight.data.fill_(init_logit)
        elif self.alpha_init == 'scaled':
            max_prec = max(self.bits)
            scaled_val = torch.tensor([bit/max_prec for bit in self.bits])
            if self.gumbel:
                scaled_prob = F.softmax(scaled_val, dim=0)
                scaled_logit = torch.log(scaled_prob/(1-scaled_prob))
            else:
                scaled_logit = scaled_val
            for i in range(len(self.bits)):
                self.alpha_weight.data[i].fill_(scaled_logit[i])
        else:
            raise ValueError(f'Unknown alpha_init: {self.alpha_init}')

        if isinstance(kwargs['kernel_size'], tuple):
            k_size = kwargs['kernel_size'][0] * kwargs['kernel_size'][1]
        else:
            k_size = kwargs['kernel_size'] * kwargs['kernel_size']

        # Quantizer
        self.mix_weight = nn.ModuleList()
        self.train_scale_param = kwargs.pop('train_scale_param', True)
        for bit in self.bits:
            self.mix_weight.append(FQConvWeightQuantization(
                outplane, k_size, num_bits=bit, train_scale_param=self.train_scale_param))

        self.conv = nn.Conv2d(inplane, outplane, **kwargs)

        if self.gumbel:
            self.register_buffer(
                'sw_buffer', torch.zeros(self.alpha_weight.shape, dtype=torch.float))

    def forward(self, input, temp, is_hard):
        mix_quant_weight = []
        if not self.gumbel:
            sw = F.softmax(self.alpha_weight/temp, dim=0)
        else:
            # If is_hard is True the output is one-hot
            if self.training:  # If model.train()
                sw = F.gumbel_softmax(self.alpha_weight, tau=temp, hard=is_hard, dim=0)
                self.sw_buffer = sw.clone().detach()
            else:  # If model.eval()
                sw = self.sw_buffer

        conv = self.conv
        weight = conv.weight
        bias = getattr(conv, 'bias', None)
        for i, bit in enumerate(self.bits):
            quant_weight = self.mix_weight[i](weight)
            scaled_quant_weight = quant_weight * sw[i].view((self.cout, 1, 1, 1))
            mix_quant_weight.append(scaled_quant_weight)

        # Quantize bias if present
        if bias is not None:
            quant_bias = _bias_sym_min_max_quantize.apply(bias, 32)
        else:
            quant_bias = bias

        # Obtain multi-precision kernel
        mix_quant_weight = sum(mix_quant_weight)
        # Compute conv
        out = F.conv2d(
            input, mix_quant_weight, quant_bias, conv.stride,
            conv.padding, conv.dilation, conv.groups)

        return out


# MR
class MultiPrecActivConv2d(nn.Module):

    def __init__(self, hw_model, inplane, outplane, wbits, abits,
                 share_weight=True, fc=None, **kwargs):
        super().__init__()
        self.hw_model = hw_model
        self.wbits = wbits
        self.abits = abits

        self.reg_target = kwargs.pop('reg_target', 'cycle')

        self.fix_qtz = kwargs.pop('fix_qtz', False)

        self.search_types = ['fixed', 'mixed', 'multi']
        if fc in self.search_types:
            self.fc = fc
        else:
            self.fc = False

        self.gumbel = kwargs.pop('gumbel', False)
        self.temp = 1

        # build mix-precision branches
        self.mix_activ = MixQuantPaCTActiv(self.abits, gumbel=self.gumbel)
        # for multiprec, only share-weight is feasible
        assert share_weight
        if not self.fc:
            self.mix_weight = SharedMultiPrecConv2d(
                inplane, outplane, self.wbits, gumbel=self.gumbel, **kwargs)
        else:
            # If the layer is fc we can use:
            if self.fc == 'fixed':
                # Fixed quantization on 8bits
                self.mix_weight = QuantMixChanConv2d(inplane, outplane, 8, **kwargs)
            elif self.fc == 'mixed':
                # Mixed-precision search
                self.mix_weight = SharedMixQuantChanConv2d(
                    inplane, outplane, wbits, gumbel=self.gumbel, **kwargs)
            elif self.fc == 'multi':
                # Multi-precision search
                self.mix_weight = SharedMultiPrecConv2d(
                    inplane, outplane, wbits, gumbel=self.gumbel, **kwargs)
            else:
                raise ValueError(f"Unknown fc search, possible values are {self.search_types}")

        # complexities
        stride = kwargs['stride'] if 'stride' in kwargs else 1
        if isinstance(kwargs['kernel_size'], tuple):
            kernel_size = kwargs['kernel_size'][0] * kwargs['kernel_size'][1]
        else:
            kernel_size = kwargs['kernel_size'] * kwargs['kernel_size']
        self.param_size = inplane * outplane * kernel_size / kwargs['groups'] * 1e-6
        self.filter_size = self.param_size / float(stride ** 2.0)
        self.register_buffer('size_product', torch.tensor(0, dtype=torch.float))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.float))

    def forward(self, input, temp, is_hard):
        self.temp = temp
        in_shape = input.shape
        tmp = torch.tensor(in_shape[1] * in_shape[2] * in_shape[3] * 1e-3, dtype=torch.float)
        self.memory_size.copy_(tmp)
        tmp = torch.tensor(self.filter_size * in_shape[-1] * in_shape[-2], dtype=torch.float)
        self.size_product.copy_(tmp)
        if not self.fix_qtz:
            out = self.mix_activ(input, temp, is_hard)
        else:
            out = _channel_asym_min_max_quantize.apply(input, 8)
        out = self.mix_weight(out, temp, is_hard)
        return out

    def complexity_loss(self):
        cout = self.mix_weight.cout
        abits = self.mix_activ.bits
        wbits = self.mix_weight.bits
        if not self.fix_qtz:
            s_a = F.softmax(self.mix_activ.alpha_activ/self.temp, dim=0)
        else:  # Layer with fixed quantization activations at the maximum available bit width
            s_a = torch.zeros(len(abits), dtype=torch.float).to(self.mix_activ.alpha_activ.device)
            s_a[-1] = 1.
        s_w = F.softmax(self.mix_weight.alpha_weight/self.temp, dim=0)

        cycles = []
        cycle = 0
        for i, bit in enumerate(wbits):
            if bit == 2:  # Analog accelerator
                cycle = sum(s_w[i]) / self.hw_model('analog')
            else:  # Digital accelerator
                cycle = sum(s_w[i]) / self.hw_model('digital')
            cycles.append(cycle)

        # Build tensor of cycles
        # NB: torch.tensor() does not preserve gradients!!!
        t_cycles = torch.stack(cycles)
        # Compute softmax
        s_c = F.softmax(t_cycles, dim=0)
        t_c = torch.dot(s_c, t_cycles)

        return t_c * self.size_product.item() / cout
        # return torch.max(torch.tensor(cycles)) * self.size_product.item() / cout

    def fetch_best_arch(self, layer_idx):
        size_product = float(self.size_product.cpu().numpy())
        memory_size = float(self.memory_size.cpu().numpy())
        if not self.fix_qtz:
            prob_activ = F.softmax(self.mix_activ.alpha_activ/self.temp, dim=0)
            prob_activ = prob_activ.detach().cpu().numpy()
            best_activ = prob_activ.argmax()
            mix_abit = 0
            abits = self.mix_activ.bits
            for i in range(len(abits)):
                mix_abit += prob_activ[i] * abits[i]
        else:
            prob_activ = 1
            best_activ = -1
            abits = self.mix_activ.bits
            mix_abit = 8
        if not self.fc or self.fc == 'multi':
            prob_weight = F.softmax(self.mix_weight.alpha_weight/self.temp, dim=0)
            prob_weight = prob_weight.detach().cpu().numpy()
            best_weight = prob_weight.argmax(axis=0)
            mix_wbit = 0
            wbits = self.mix_weight.bits
            cout = self.mix_weight.cout
            for i in range(len(wbits)):
                mix_wbit += sum(prob_weight[i]) * wbits[i]
            mix_wbit = mix_wbit / cout
        else:
            if self.fc == 'fixed':
                prob_weight = 1
                mix_wbit = 8
            elif self.fc == 'mixed':
                prob_weight = F.softmax(self.mix_weight.alpha_weight/self.temp, dim=0)
                prob_weight = prob_weight.detach().cpu().numpy()
                best_weight = prob_weight.argmax(axis=0)
                mix_wbit = 0
                wbits = self.mix_weight.bits
                for i in range(len(wbits)):
                    mix_wbit += prob_weight[i] * wbits[i]

        weight_shape = list(self.mix_weight.conv.weight.shape)
        print('idx {} with shape {}, activ alpha: {}, comp: {:.3f}M * {:.3f} * {:.3f}, '
              'memory: {:.3f}K * {:.3f}'.format(layer_idx, weight_shape, prob_activ, size_product,
                                                mix_abit, mix_wbit, memory_size, mix_abit))
        print('idx {} with shape {}, weight alpha: {}, comp: {:.3f}M * {:.3f} * {:.3f}, '
              'param: {:.3f}M * {:.3f}'.format(layer_idx, weight_shape, prob_weight, size_product,
                                               mix_abit, mix_wbit, self.param_size, mix_wbit))
        if not self.fc or self.fc == 'multi':
            best_wbit = sum([wbits[_] for _ in best_weight]) / cout
            eff_mac_cycles = []
            eff_mac_cycle = 0
            for i, bit in enumerate(wbits):
                if bit == 2:
                    eff_mac_cycle = sum(best_weight == i) / (self.hw_model('analog') * cout)
                else:
                    eff_mac_cycle = sum(best_weight == i) / (self.hw_model('digital') * cout)
                eff_mac_cycles.append(eff_mac_cycle)
            slowest_eff_mac_cycle = max(eff_mac_cycles)
        else:
            if self.fc == 'fixed':
                best_wbit = 8
                best_weight = 8
                # mac_cycle = self.hw_model('digital')
            elif self.fc == 'mixed':
                best_wbit = wbits[best_weight]
                # mac_cycle = self.hw_model('digital')

        if not self.fix_qtz:
            best_arch = {'best_activ': [best_activ], 'best_weight': [best_weight]}
            # bitops = size_product * abits[best_activ] * best_wbit
            bita = memory_size * abits[best_activ]
        else:
            best_arch = {'best_activ': [8], 'best_weight': [best_weight]}
            # bitops = size_product * 8 * best_wbit
            bita = memory_size * 8

        bitw = self.param_size * best_wbit
        cycles = size_product * slowest_eff_mac_cycle
        mixbitops = size_product * mix_abit * mix_wbit
        mixbita = memory_size * mix_abit
        mixbitw = self.param_size * mix_wbit

        return best_arch, cycles, bita, bitw, mixbitops, mixbita, mixbitw
        # return best_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw
