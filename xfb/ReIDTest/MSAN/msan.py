import sys

sys.path.append(".")

import torch
from torch import nn
import torch.nn.functional as F

from fastreid.layers import *
from fastreid.modeling import BaseModule
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.heads.embedding_head import EmbeddingHead
from fastreid.modeling.losses import *
from fastreid.modeling.meta_arch.build import META_ARCH_REGISTRY
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier

import torchsnooper


class MSANEmbeddingHead(nn.Module):
    """
    code reference:
        fastreid/modeling/heads/embedding_head.py->EmbeddingHead()
    """

    def __init__(self, cfg, feat_dim):
        super().__init__()
        # fmt: off
        embedding_dim = cfg.MODEL.HEADS.EMBEDDING_DIM
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_TYPE
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER
        with_bnneck   = cfg.MODEL.HEADS.WITH_BNNECK
        norm_type     = cfg.MODEL.HEADS.NORM
        # fmt: on

        valid_pool_type = (
            "fastavgpool",
            "avgpool",
            "maxpool",
            "gempoolP",
            "gempool",
            "avgmaxpool",
            "clipavgpool",
            "identity",
            "flatten",
        )
        if not set(pool_type).issubset(valid_pool_type):
            raise KeyError(f"{set(pool_type)-set(valid_pool_type)} is not supported!")

        self.pool_layer = []
        if "fastavgpool" in pool_type:
            self.pool_layer.append(FastGlobalAvgPool2d())
        if "avgpool" in pool_type:
            self.pool_layer.append(nn.AdaptiveAvgPool2d(1))
        if "maxpool" in pool_type:
            self.pool_layer.append(nn.AdaptiveMaxPool2d(1))
        if "gempoolP" in pool_type:
            self.pool_layer.append(GeneralizedMeanPoolingP())
        if "gempool" in pool_type:
            self.pool_layer.append(GeneralizedMeanPooling())
        if "avgmaxpool" in pool_type:
            self.pool_layer.append(AdaptiveAvgMaxPool2d())
        if "clipavgpool" in pool_type:
            self.pool_layer.append(ClipGlobalAvgPool2d())
        if "identity" in pool_type:
            self.pool_layer.append(nn.Identity())
        if "flatten" in pool_type:
            self.pool_layer.append(Flatten())

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

    def forward(self, features, targets=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        pool_feats = [pool(features) for pool in self.pool_layer]
        global_feat = sum(pool_feats)
        bn_feat = self.bottleneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]

        # Evaluation
        if not self.training:
            return bn_feat

        # Training
        if self.classifier.__class__.__name__ == "Linear":
            cls_outputs = self.classifier(bn_feat)
            pred_class_logits = F.linear(bn_feat, self.classifier.weight)
        else:
            cls_outputs = self.classifier(bn_feat, targets)
            pred_class_logits = self.classifier.s * F.linear(
                F.normalize(bn_feat), F.normalize(self.classifier.weight)
            )

        # fmt: off
        if self.neck_feat == "before":  feat = global_feat[..., 0, 0]
        elif self.neck_feat == "after": feat = bn_feat
        else:                           raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
        # fmt: on

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": pred_class_logits,
            "pool_feats": pool_feats,
            "features": feat,
        }


@META_ARCH_REGISTRY.register()
class MSAN(BaseModule):
    """
    Reference:
        Person Re-identification Based on Multi-scale and Attention Fusion
        Journal of Electronics & Information Technology 2020
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.is_multi_scale = cfg.MODEL.MULTI_SCALE
        self.use_all_scale_feat = cfg.MODEL.USE_ALL_SCALE_FEAT

        # backbone
        backbone = build_backbone(cfg)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3[0],
        )
        if self.is_multi_scale:
            self.b1_head = MSANEmbeddingHead(cfg, 1024)  # from res_conv4a

        self.res_conv4x = backbone.layer3[1:]  # res_conv4b ~ res_conv4f
        if self.is_multi_scale:
            self.b2_head = MSANEmbeddingHead(cfg, 1024)  # from res_conv4x

        self.res_conv5x = backbone.layer4  # res_conv5a ~ res_conv5c
        self.b3_head = MSANEmbeddingHead(cfg, 2048)

    # @torchsnooper.snoop()
    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)  # (bs, 3, 256, 128)
        res4a_feat = self.backbone(images)  # (bs, 1024, 16, 8)
        res4f_feat = self.res_conv4x(res4a_feat)  # (bs, 1024, 16, 8)
        res5c_feat = self.res_conv5x(res4f_feat)  # (bs, 2048, 16, 8)

        if self.training:
            assert (
                "targets" in batched_inputs
            ), "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].long().to(self.device)
            if targets.sum() < 0:
                targets.zero_()

            b3_outputs = self.b3_head(res5c_feat, targets)

            if self.is_multi_scale:
                b1_outputs = self.b1_head(res4a_feat, targets)
                b2_outputs = self.b2_head(res4f_feat, targets)
                return {
                    "b1_outputs": b1_outputs,
                    "b2_outputs": b2_outputs,
                    "b3_outputs": b3_outputs,
                    "targets": targets,
                }
            else:
                return {
                    "b3_outputs": b3_outputs,
                    "targets": targets,
                }
        else:
            b3_feat = self.b3_head(res5c_feat)

            if self.use_all_scale_feat:
                b1_feat = self.b1_head(res4a_feat)
                b2_feat = self.b2_head(res4f_feat)
                united_feat = torch.cat(
                    (b1_feat, b2_feat, b3_feat), dim=1
                )  # (bs, 4096)
                united_feat[:, :1024] = F.normalize(united_feat[:, :1024], dim=1)
                united_feat[:, 1024:2048] = F.normalize(
                    united_feat[:, 1024:2048], dim=1
                )
                united_feat[:, 2048:] = F.normalize(united_feat[:, 2048:], dim=1)
                return united_feat
            else:
                return b3_feat

    def losses(self, outs):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # fmt: off
        gt_labels = outs["targets"]

        if self.is_multi_scale:
            b1_outputs        = outs["b1_outputs"]
            b2_outputs        = outs["b2_outputs"]

            # model predictions
            b1_pred_class_logits = b1_outputs['pred_class_logits'].detach()
            b1_logits            = b1_outputs['cls_outputs']
            b1_pool_feat         = b1_outputs['features']
            b2_pred_class_logits = b2_outputs['pred_class_logits'].detach()
            b2_logits            = b2_outputs['cls_outputs']
            b2_pool_feat         = b2_outputs['features']

            # Log prediction accuracy
            self.log_accuracy(b1_pred_class_logits, gt_labels, "b1_cls_acc")
            self.log_accuracy(b2_pred_class_logits, gt_labels, "b2_cls_acc")

        b3_outputs = outs["b3_outputs"]
        # model predictions
        b3_pred_class_logits = b3_outputs['pred_class_logits'].detach()
        b3_logits            = b3_outputs['cls_outputs']
        b3_pool_feat         = b3_outputs['features']
        # fmt: on

        # Log prediction accuracy
        self.log_accuracy(b3_pred_class_logits, gt_labels, "b3_cls_acc")

        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        if self.is_multi_scale:
            if "CrossEntropyLoss" in loss_names:
                loss_dict["loss_cls_b1"] = (
                    cross_entropy_loss(
                        b1_logits,
                        gt_labels,
                        self._cfg.MODEL.LOSSES.CE.EPSILON,
                        self._cfg.MODEL.LOSSES.CE.ALPHA,
                    )
                    * self._cfg.MODEL.LOSSES.CE.SCALE
                    / 3.0
                )
                loss_dict["loss_cls_b2"] = (
                    cross_entropy_loss(
                        b2_logits,
                        gt_labels,
                        self._cfg.MODEL.LOSSES.CE.EPSILON,
                        self._cfg.MODEL.LOSSES.CE.ALPHA,
                    )
                    * self._cfg.MODEL.LOSSES.CE.SCALE
                    / 3.0
                )
                loss_dict["loss_cls_b3"] = (
                    cross_entropy_loss(
                        b3_logits,
                        gt_labels,
                        self._cfg.MODEL.LOSSES.CE.EPSILON,
                        self._cfg.MODEL.LOSSES.CE.ALPHA,
                    )
                    * self._cfg.MODEL.LOSSES.CE.SCALE
                    / 3.0
                )

        if "TripletLoss" in loss_names:
            loss_dict["loss_triplet_b1"] = (
                triplet_loss(
                    b1_pool_feat,
                    gt_labels,
                    self._cfg.MODEL.LOSSES.TRI.MARGIN,
                    self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                    self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
                )
                * self._cfg.MODEL.LOSSES.TRI.SCALE
                / 3.0
            )
            loss_dict["loss_triplet_b2"] = (
                triplet_loss(
                    b2_pool_feat,
                    gt_labels,
                    self._cfg.MODEL.LOSSES.TRI.MARGIN,
                    self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                    self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
                )
                * self._cfg.MODEL.LOSSES.TRI.SCALE
                / 3.0
            )
            loss_dict["loss_triplet_b3"] = (
                triplet_loss(
                    b3_pool_feat,
                    gt_labels,
                    self._cfg.MODEL.LOSSES.TRI.MARGIN,
                    self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                    self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
                )
                * self._cfg.MODEL.LOSSES.TRI.SCALE
                / 3.0
            )
        else:
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


if __name__ == "__main__":
    import IPython
    from fastreid.config import cfg

    cfg.MODEL.BACKBONE.WITH_CBAM = True
    cfg.MODEL.BACKBONE.PRETRAIN_PATH = (
        "/home/xfb/.cache/torch/checkpoints/resnet50-19c8e357.pth"
    )
    model = MSAN(cfg)
    x = torch.randn(10, 3, 256, 128)
    y = model(x)
    # print(model.res_conv4x)
    IPython.embed()
    print(model)
