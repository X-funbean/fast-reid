import sys

sys.path.append(".")

from fastreid.config import get_cfg
from fastreid.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from fastreid.utils.checkpoint import Checkpointer

from config import add_attr_config
from msan import MSAN

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_attr_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    trainer = DefaultTrainer

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        # model = trainer.build_model(cfg)
        model = MSAN(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

        res = trainer.test(cfg, model)
        return res

    trainer = trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
