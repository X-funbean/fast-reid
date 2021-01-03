from fastreid.config import CfgNode as CN


def add_attr_config(cfg):
    _C = cfg

    _C.MODEL.MULTI_SCALE = True
    _C.MODEL.HEADS.POOL_TYPE = (
        # "avgpool",
        # "maxpool",
        "avgmaxpool",
    )
    _C.MODEL.USE_ALL_SCALE_FEAT = True 

    _C.TRAINER = CN()
    _C.TRAINER.DEFAULT = True