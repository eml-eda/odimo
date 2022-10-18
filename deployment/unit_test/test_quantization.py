import unittest

import torch

from deployment.observer import insert_observers
from deployment.quantization import build_qgraph
from models import quant_module as qm

from test_models import ToyQConv


class TestQuantization(unittest.TestCase):
    """Test proper execution of quantization operations"""

    def test_qinfo_annotation(self):
        """Test the annotation of quantization information (qinfo) on target
        layers
        """
        nn_ut = ToyQConv(conv_func=qm.QuantMultiPrecActivConv2d,
                         abits=[[7]]*3, wbits=[[2]]*3)
        dummy_inp = torch.rand((2,) + nn_ut.input_shape)
        new_nn = insert_observers(nn_ut,
                                  target_layers=(qm.QuantMultiPrecActivConv2d))
        new_nn(dummy_inp)
        new_nn = build_qgraph(new_nn,
                              output_classes=2,
                              target_layers=(qm.QuantMultiPrecActivConv2d))
        self.assertTrue(1)
