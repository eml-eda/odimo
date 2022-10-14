import torch.nn as nn
import torch.nn.functional as F


class ToyFPConv(nn.Module):

    def __init__(self, n=2):
        super().__init__()
        self.input_shape = (1, 32, 32)
        self.net = nn.ModuleDict(
            {f'conv{idx}': nn.Conv2d(idx, idx+1, 3, padding=1)
             for idx in range(1, n+1)}
        )

    def forward(self, x):
        for idx, conv in enumerate(self.net.values()):
            if idx == 0:
                x = conv(x)
            else:
                x = conv(F.relu(x))
        return x


class ToyQConv(nn.Module):

    def __init__(self, conv_func):
        super().__init__()
        self.input_shape = (1, 32, 32)
        self.conv0 = conv_func()
        self.conv1 = conv_func()

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        return x1
