import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

__all__ = [
    'FakeIntMultiPrecActivConv2d',
]


class ClippedLinearQuantizeSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, num_bits, act_scale):
        clip_val = (2**num_bits - 1) * act_scale
        out = torch.clamp(x, 0, clip_val.data[0])
        # out = torch.max(torch.min(x, clip_val), torch.tensor(0.))
        out_q = torch.floor(out / act_scale)
        return out_q

    @staticmethod
    def backward(ctx, grad_output):
        # Need to do something here??
        return grad_output, None, None


class ClippedLinearQuantization(nn.Module):

    def __init__(self, num_bits, act_scale):
        super().__init__()
        self.num_bits = num_bits[0]
        self.act_scale = act_scale

    def forward(self, x):
        x_q = ClippedLinearQuantizeSTE.apply(x, self.num_bits, self.act_scale)
        return x_q


class IntPaCTActiv(nn.Module):

    def __init__(self, abits, act_scale):
        super().__init__()
        self.abits = abits
        self.act_scale = act_scale

        self.quantizer = ClippedLinearQuantization(num_bits=abits,
                                                   act_scale=act_scale)

    def forward(self, x):
        x_q = self.quantizer(x)
        return x_q


class IntMultiPrecActivConv2d(nn.Module):

    def __init__(self, int_params, abits, wbits, **kwargs):
        super().__init__()
        self.act_scale = int_params['s_x']
        self.abits = abits
        self.wbits = wbits
        self.first_layer = int_params['first']

        self.mix_activ = IntPaCTActiv(abits, self.act_scale)
        self.mix_weight = IntMultiPrecConv2d(int_params, wbits, **kwargs)
        # self.conv = nn.Conv2d(**kwargs)

    def forward(self, act_in):
        if self.first_layer:
            act_q = self.mix_activ(act_in)
        else:
            # act_q = torch.floor(F.relu(act_in))
            act_q = torch.floor(
                torch.min(  # ReLU127
                    torch.max(torch.tensor(0.), act_in),
                    torch.tensor(127.))
            )
        act_out = self.mix_weight(act_q)
        return act_out


class IntMultiPrecConv2d(nn.Module):

    def __init__(self, int_params, wbits, **kwargs):
        super().__init__()
        self.bits = wbits

        self.b_16 = int_params['b_16']
        self.n_sh = int_params['n_sh']
        if wbits == [2]:
            self.alpha = int_params['alpha']
            self.b_8 = int_params['b_8']
            self.n_b = int_params['n_b']

        self.cout = kwargs['out_channels']

        self.alpha_weight = Parameter(torch.Tensor(len(self.bits), self.cout),
                                      requires_grad=False)
        self.alpha_weight.data.fill_(0.01)

        self.conv = nn.Conv2d(**kwargs)

    def forward(self, x):
        out = []
        sw = F.one_hot(torch.argmax(self.alpha_weight, dim=0),
                       num_classes=len(self.bits)).t()
        conv = self.conv
        weight = conv.weight
        for i, bit in enumerate(self.bits):
            eff_weight = weight * sw[i].view((self.cout, 1, 1, 1))
            if bit == 2:
                conv_out = F.conv2d(
                    x, eff_weight, None, conv.stride,
                    conv.padding, conv.dilation, conv.groups)
                alpha = self.alpha.view(1, self.cout, 1, 1)
                b = (self.b_8 * 2**self.n_b).view(1, self.cout, 1, 1)
                scale_out = alpha * conv_out + b
                shift = (2**self.n_sh).view(1, self.cout, 1, 1)
                out.append(scale_out / shift)
            elif bit == 8:
                if self.b_16 is not None:
                    eff_bias = self.b_16 * sw[i].view(self.cout)
                else:
                    eff_bias = None
                conv_out = F.conv2d(
                    x, eff_weight, eff_bias, conv.stride,
                    conv.padding, conv.dilation, conv.groups)
                out.append(conv_out / 2**self.n_sh)
        out = sum(out)

        return out


class FakeIntMultiPrecActivConv2d(nn.Module):

    def __init__(self, act_scale, abits, wbits, **kwargs):
        super().__init__()
        self.act_scale = act_scale
        self.abits = abits
        self.wbits = wbits

        self.mix_activ = IntPaCTActiv(abits, act_scale)

        self.conv = nn.Conv2d(**kwargs)

    def forward(self, act_in):
        act_q = self.mix_activ(act_in)
        act_out = self.conv(act_q)
        return act_out
