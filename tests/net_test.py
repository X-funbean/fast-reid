import unittest

import torch

import sys
sys.path.append('.')
from fastreid.config import cfg
from fastreid.modeling.backbones import build_resnet_backbone, build_hrnet_backbone
from torch import nn


class MyTestCase(unittest.TestCase):
    def test_hrnet_v2(self):
        net1 = build_hrnet_backbone(cfg)
        net1.cuda()
        
        assert y1.sum() == y2.sum(), 'train mode problem'
        net1.eval()
        net2.eval()
        y1 = net1(x)
        y2 = net2(x)
        assert y1.sum() == y2.sum(), 'eval mode problem'


if __name__ == '__main__':
    unittest.main()
