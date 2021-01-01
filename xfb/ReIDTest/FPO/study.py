import sys
sys.path.append('.')

import torch
from torch import nn
from torchvision.models import resnet50

from fastreid.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads.build import build_heads

from fpo.fpo import FPO

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

if __name__ == "__main__":
    from fastreid.config import get_cfg
    args = default_argument_parser().parse_args()
    cfg = setup(args)

    model = FPO(cfg)
    print(model)