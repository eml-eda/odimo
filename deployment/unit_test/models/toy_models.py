import torch.nn as nn


class ToySequentialConv(nn.Module):

    def __init__(self, conv_func):
        super().__init__()
        self.input_shape = (1, 32, 32)
        self.conv0 = conv_func()
        self.conv1 = conv_func()

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        return x1
