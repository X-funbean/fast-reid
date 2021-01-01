from fastreid.config import CfgNode as CN


def add_fpo_config(cfg):
    _C = cfg

    # ---------------------------------------------------------------------------- #
    # res4f b1 Optimizer
    # ---------------------------------------------------------------------------- #
    # _C.MODEL.LOSSES.BCE = CN()
    # _C.MODEL.LOSSES.BCE.WEIGHT_ENABLED = True
    # _C.MODEL.LOSSES.BCE.SCALE = 1.0

    # _C.TEST.THRES = 0.5