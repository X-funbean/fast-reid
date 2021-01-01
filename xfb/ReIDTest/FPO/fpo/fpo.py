import copy
from fastreid.modeling.heads.embedding_head import EmbeddingHead

import torch
from torch import nn
import torch.nn.functional as F

from fastreid.layers import *
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads.build import build_heads, REID_HEADS_REGISTRY
from fastreid.modeling.losses import *
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY
from fastreid.utils.events import get_event_storage
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier


class FPOEmbeddingHead(EmbeddingHead):
    def __init__(self, cfg, feat_dim):
        super().__init__(cfg)
        # fmt: off
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER
        with_bnneck   = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type     = cfg.MODEL.HEADS.NORM

        if pool_type == 'fastavgpool':   self.pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':     self.pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'maxpool':     self.pool_layer = nn.AdaptiveMaxPool2d(1)
        elif pool_type == 'gempoolP':    self.pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == 'gempool':     self.pool_layer = GeneralizedMeanPooling()
        elif pool_type == "avgmaxpool":  self.pool_layer = AdaptiveAvgMaxPool2d()
        elif pool_type == 'clipavgpool': self.pool_layer = ClipGlobalAvgPool2d()
        elif pool_type == "identity":    self.pool_layer = nn.Identity()
        elif pool_type == "flatten":     self.pool_layer = Flatten()
        else:                            raise KeyError(f"{pool_type} is not supported!")
        # fmt: on

        self.neck_feat = neck_feat

        bottleneck = []
        if embedding_dim > 0:
            bottleneck.append(nn.Conv2d(feat_dim, embedding_dim, 1, 1, bias=False))
            feat_dim = embedding_dim

        if with_bnneck:
            bottleneck.append(get_norm(norm_type, feat_dim, bias_freeze=True))

        self.bottleneck = nn.Sequential(*bottleneck)

        # identity classification layer
        # fmt: off
        if cls_type == 'linear':          self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        elif cls_type == 'arcSoftmax':    self.classifier = ArcSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'circleSoftmax': self.classifier = CircleSoftmax(cfg, feat_dim, num_classes)
        elif cls_type == 'cosSoftmax':    self.classifier = CosSoftmax(cfg, feat_dim, num_classes)
        else:                             raise KeyError(f"{cls_type} is not supported!")
        # fmt: on

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)


@META_ARCH_REGISTRY.register()
class FPO(nn.Module):
    """
    Reference:
        Person Re-Identification with Feature Pyramid Optimization and Gradual Background Suppression
        Neural Networks 2020
    """

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

        # backbone
        backbone = build_backbone(cfg)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )

        self.res_conv5 = backbone.layer4

        # branch1: res4f
        self.b1_head = FPOEmbeddingHead(cfg, 1024)

        # branch2: res5c
        self.b2_head = FPOEmbeddingHead(cfg, 2048)

        # branch3: united
        self.b3_head = FPOEmbeddingHead(cfg, 3072)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)  # (bs, 3, 256, 128)
        res4f_feat = self.backbone(images)  # (bs, 1024, 16, 8)
        res5c_feat = self.res_conv5(res4f_feat)  # (bs, 2048, 16, 8)
        united_feat = torch.cat((res4f_feat, res5c_feat), dim=1)  # (bs, 3072, 16, 8)

        if self.training:
            assert (
                "targets" in batched_inputs
            ), "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].long().to(self.device)
            if targets.sum() < 0:
                targets.zero_()

            b1_outputs = self.b1_head(res4f_feat, targets)
            b2_outputs = self.b2_head(res5c_feat, targets)
            b3_outputs = self.b3_head(united_feat, targets)

            return {
                "b1_outputs": b1_outputs,
                "b2_outputs": b2_outputs,
                "b3_outputs": b3_outputs,
                "targets": targets,
            }
        else: 
            # b2_outputs = self.b2_head(res5c_feat)
            b3_outputs = self.b3_head(united_feat) # (128, 512)
            # print("b3_outputs.shape", b3_outputs.shape)
            return b3_outputs

            # print("united_feat.shape", united_feat.shape)
            # united_feat[:, 1024:] = F.normalize(united_feat[:, 1024:], dim=1)
            # united_feat[:, :1024] = F.normalize(united_feat[:, :1024], dim=1)
            # return united_feat

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

    
    def losses(self, outs):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # fmt: off
        b1_outputs        = outs["b1_outputs"]
        b2_outputs        = outs["b2_outputs"]
        b3_outputs        = outs["b3_outputs"]
        gt_labels         = outs["targets"]
        # model predictions
        b1_pred_class_logits = b1_outputs['pred_class_logits'].detach()
        b1_logits            = b1_outputs['cls_outputs']
        b1_pool_feat         = b1_outputs['features']
        b2_pred_class_logits = b2_outputs['pred_class_logits'].detach()
        b2_logits            = b2_outputs['cls_outputs']
        b2_pool_feat         = b2_outputs['features']
        b3_pred_class_logits = b3_outputs['pred_class_logits'].detach()
        b3_logits            = b3_outputs['cls_outputs']
        b3_pool_feat         = b3_outputs['features']
        # fmt: on

        # Log prediction accuracy
        self.log_accuracy(b1_pred_class_logits, gt_labels, "b1_cls_acc")
        self.log_accuracy(b2_pred_class_logits, gt_labels, "b2_cls_acc")
        self.log_accuracy(b3_pred_class_logits, gt_labels, "b3_cls_acc")

        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        if "CrossEntropyLoss" in loss_names:
            loss_dict["loss_cls_b3"] = (
                cross_entropy_loss(
                    b3_logits,
                    gt_labels,
                    self._cfg.MODEL.LOSSES.CE.EPSILON,
                    self._cfg.MODEL.LOSSES.CE.ALPHA,
                )
                * self._cfg.MODEL.LOSSES.CE.SCALE
                
            )

        if "TripletLoss" in loss_names:
            loss_dict["loss_triplet_b3"] = (
                triplet_loss(
                    b3_pool_feat,
                    gt_labels,
                    self._cfg.MODEL.LOSSES.TRI.MARGIN,
                    self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                    self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
                )
                * self._cfg.MODEL.LOSSES.TRI.SCALE
                
            )

        return loss_dict

    # def losses(self, outs):
    #     r"""
    #     Compute loss from modeling's outputs, the loss function input arguments
    #     must be the same as the outputs of the model forwarding.
    #     """
    #     # fmt: off
    #     b1_outputs        = outs["b1_outputs"]
    #     b2_outputs        = outs["b2_outputs"]
    #     b3_outputs        = outs["b3_outputs"]
    #     gt_labels         = outs["targets"]
    #     # model predictions
    #     b1_pred_class_logits = b1_outputs['pred_class_logits'].detach()
    #     b1_logits            = b1_outputs['cls_outputs']
    #     b1_pool_feat         = b1_outputs['features']
    #     b2_pred_class_logits = b2_outputs['pred_class_logits'].detach()
    #     b2_logits            = b2_outputs['cls_outputs']
    #     b2_pool_feat         = b2_outputs['features']
    #     b3_pred_class_logits = b3_outputs['pred_class_logits'].detach()
    #     b3_logits            = b3_outputs['cls_outputs']
    #     b3_pool_feat         = b3_outputs['features']
    #     # fmt: on

    #     # Log prediction accuracy
    #     self.log_accuracy(b1_pred_class_logits, gt_labels, "b1_cls_acc")
    #     self.log_accuracy(b2_pred_class_logits, gt_labels, "b2_cls_acc")
    #     self.log_accuracy(b3_pred_class_logits, gt_labels, "b3_cls_acc")

    #     loss_dict = {}
    #     loss_names = self._cfg.MODEL.LOSSES.NAME

    #     if "CrossEntropyLoss" in loss_names:
    #         loss_dict["loss_cls_b1"] = (
    #             cross_entropy_loss(
    #                 b1_logits,
    #                 gt_labels,
    #                 self._cfg.MODEL.LOSSES.CE.EPSILON,
    #                 self._cfg.MODEL.LOSSES.CE.ALPHA,
    #             )
    #             * self._cfg.MODEL.LOSSES.CE.SCALE
    #             / 3.0
    #         )
    #         loss_dict["loss_cls_b2"] = (
    #             cross_entropy_loss(
    #                 b2_logits,
    #                 gt_labels,
    #                 self._cfg.MODEL.LOSSES.CE.EPSILON,
    #                 self._cfg.MODEL.LOSSES.CE.ALPHA,
    #             )
    #             * self._cfg.MODEL.LOSSES.CE.SCALE
    #             / 3.0
    #         )
    #         loss_dict["loss_cls_b3"] = (
    #             cross_entropy_loss(
    #                 b3_logits,
    #                 gt_labels,
    #                 self._cfg.MODEL.LOSSES.CE.EPSILON,
    #                 self._cfg.MODEL.LOSSES.CE.ALPHA,
    #             )
    #             * self._cfg.MODEL.LOSSES.CE.SCALE
    #             / 3.0
    #         )

    #     if "TripletLoss" in loss_names:
    #         loss_dict["loss_triplet_b1"] = (
    #             triplet_loss(
    #                 b1_pool_feat,
    #                 gt_labels,
    #                 self._cfg.MODEL.LOSSES.TRI.MARGIN,
    #                 self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
    #                 self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
    #             )
    #             * self._cfg.MODEL.LOSSES.TRI.SCALE
    #             / 3.0
    #         )
    #         loss_dict["loss_triplet_b2"] = (
    #             triplet_loss(
    #                 b2_pool_feat,
    #                 gt_labels,
    #                 self._cfg.MODEL.LOSSES.TRI.MARGIN,
    #                 self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
    #                 self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
    #             )
    #             * self._cfg.MODEL.LOSSES.TRI.SCALE
    #             / 3.0
    #         )
    #         loss_dict["loss_triplet_b3"] = (
    #             triplet_loss(
    #                 b3_pool_feat,
    #                 gt_labels,
    #                 self._cfg.MODEL.LOSSES.TRI.MARGIN,
    #                 self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
    #                 self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
    #             )
    #             * self._cfg.MODEL.LOSSES.TRI.SCALE
    #             / 3.0
    #         )

    #     return loss_dict

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
