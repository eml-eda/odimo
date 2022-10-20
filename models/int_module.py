import torch
import torch.nn as nn

__all__ = [
    'FakeIntMultiPrecActivConv2d',
]


class ClippedLinearQuantizeSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, num_bits, act_scale):
        clip_val = (2**num_bits - 1) * act_scale
        # out = torch.clamp(x, 0, clip_val)
        out = torch.max(torch.min(x, clip_val), 0.)
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


class FakeIntPaCTActiv(nn.Module):

    def __init__(self, abits, act_scale):
        super().__init__()
        self.abits = abits
        self.act_scale = act_scale

        self.quantizer = ClippedLinearQuantization(num_bits=abits,
                                                   act_scale=act_scale)

    def forward(self, x):
        x_q = self.quantizer(x)
        return x_q


class FakeIntMultiPrecActivConv2d(nn.Module):

    def __init__(self, act_scale, abits, wbits, **kwargs):
        super().__init__()
        self.act_scale = act_scale
        self.abits = abits
        self.wbits = wbits

        self.mix_activ = FakeIntPaCTActiv(abits, act_scale)

        self.conv = nn.Conv2d(**kwargs)

    def forward(self, act_in):
        act_q = self.mix_activ(act_in)
        act_out = self.conv(act_q)
        return act_out
