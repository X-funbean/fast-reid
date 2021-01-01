import torch
from torch import nn

from ..utils.events import get_event_storage


class BaseModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer(
            "pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1)
        )

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs):
        r"""
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        else:
            raise TypeError(
                "batched_inputs must be dict or torch.Tensor, but get {}".format(
                    type(batched_inputs)
                )
            )

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    @staticmethod
    def log_accuracy(pred_class_logits, gt_classes, name="cls_accuracy", topk=(1,)):
        """
        Log the accuracy metrics to EventStorage.
        modified from fastreid/modeling/losses/cross_entropy_loss.py
        """
        bsz = pred_class_logits.size(0)
        maxk = max(topk)
        _, pred_class = pred_class_logits.topk(maxk, 1, True, True)
        pred_class = pred_class.t()
        correct = pred_class.eq(gt_classes.view(1, -1).expand_as(pred_class))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1.0 / bsz))

        storage = get_event_storage()
        storage.put_scalar(name, ret[0])