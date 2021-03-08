"""
Code references:
    https://github.com/HRNet/HRNet-Image-Classification/blob/master/lib/models/cls_hrnet.py
    https://github.com/HRNet/HRNet-Object-Detection/blob/master/configs/hrnet/faster_rcnn_hrnetv2p_w32_syncbn_mstrain_1x.py
    fastreid/modeling/backbones/resnet.py
    fastreid/modeling/backbones/osnet.py
"""

import logging

import torch
from torch import nn
import torch.nn.functional as F
import torchsnooper

from fastreid.layers import MS_CAM
from fastreid.utils import comm
from fastreid.utils.checkpoint import (
    get_missing_parameters_message,
    get_unexpected_parameters_message,
)
from .build import BACKBONE_REGISTRY

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
model_url = "https://github.com/HRNet/HRNet-Image-Classification"
model_urls = {
    "hrnetv2_w18": None,
    "hrnetv2_w32": None,
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(
            self,
            num_branches,
            blocks,
            num_blocks,
            num_inchannels,
            num_channels,
            fuse_method,
            multi_scale_output=True,
    ):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _check_branches(
            self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if (
                stride != 1
                or self.num_inchannels[branch_index]
                != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(self.num_inchannels[branch_index], num_channels[branch_index])
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3, momentum=BN_MOMENTUM
                                    ),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(
                                        num_outchannels_conv3x3, momentum=BN_MOMENTUM
                                    ),
                                    nn.ReLU(False),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    # @torchsnooper.snoop()
    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class HRNet(nn.Module):
    def __init__(
            self,
            cfg,
            stage_configs,
    ):
        super().__init__()
        self.cfg = cfg
        self.stage_configs = stage_configs

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # stage 1
        stage1_config = stage_configs[0]
        self.stage1_cfg = stage1_config
        block = blocks_dict[stage1_config["BLOCK"]]
        num_blocks = stage1_config["NUM_BLOCKS"][0]
        num_channels = stage1_config["NUM_CHANNELS"][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        # stage 2
        stage2_config = stage_configs[1]
        self.stage2_cfg = stage2_config
        block = blocks_dict[stage2_config["BLOCK"]]
        num_channels = stage2_config["NUM_CHANNELS"]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels
        )
        self.stage2, pre_stage_channels = self._make_stage(stage2_config, num_channels)

        # stage 3
        stage3_config = stage_configs[2]
        self.stage3_cfg = stage3_config
        block = blocks_dict[stage3_config["BLOCK"]]
        num_channels = stage3_config["NUM_CHANNELS"]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(stage3_config, num_channels)

        # stage 4
        stage4_config = stage_configs[3]
        self.stage4_cfg = stage4_config
        block = blocks_dict[stage4_config["BLOCK"]]
        num_channels = stage4_config["NUM_CHANNELS"]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            stage4_config, num_channels, multi_scale_output=True
        )

        if cfg.MODEL.BACKBONE.HRNET.HEAD_TYPE == "classification":
            self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(pre_stage_channels)
        elif cfg.MODEL.BACKBONE.HRNET.HEAD_TYPE == "V2":
            final_inp_channels = pre_stage_channels
            self.head = nn.Sequential(
                nn.Conv2d(
                    in_channels=final_inp_channels,
                    out_channels=final_inp_channels,
                    kernel_size=1,
                    stride=1,
                    padding=1 if cfg.MODEL.BACKBONE.HRNET.FINAL_CONV_KERNEL == 3 else 0,
                ),
                nn.BatchNorm2d(final_inp_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=final_inp_channels,
                    out_channels=cfg.MODEL.BACKBONE.FEAT_DIM,
                    kernel_size=cfg.MODEL.BACKBONE.HRNET.FINAL_CONV_KERNEL,
                    stride=1,
                    padding=1 if cfg.MODEL.BACKBONE.HRNET.FINAL_CONV_KERNEL == 3 else 0,
                ),
            )

        self.random_init()

    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(
                head_block, channels, head_channels[i], 1, stride=1
            )
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(
                                num_channels_cur_layer[i], momentum=BN_MOMENTUM
                            ),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, stage_config, num_inchannels, multi_scale_output=True):
        num_modules = stage_config["NUM_MODULES"]
        num_branches = stage_config["NUM_BRANCHES"]
        num_blocks = stage_config["NUM_BLOCKS"]
        num_channels = stage_config["NUM_CHANNELS"]
        block = blocks_dict[stage_config["BLOCK"]]
        fuse_method = stage_config["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # stage 1
        x = self.layer1(x)

        # stage 2
        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        # stage 3
        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        # stage 4
        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # head
        if self.cfg.MODEL.BACKBONE.HRNET.HEAD_TYPE == "classification":
            y = self.incre_modules[0](y_list[0])
            for i in range(len(self.downsamp_modules)):
                y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsamp_modules[i](y)
            y = self.final_layer(y)
            return y
        elif self.cfg.MODEL.BACKBONE.HRNET.HEAD_TYPE == "V2":
            x = y_list
            height, width = x[0].size(2), x[0].size(3)
            x1 = F.interpolate(x[1], size=(height, width), mode='bilinear', align_corners=False)
            x2 = F.interpolate(x[2], size=(height, width), mode='bilinear', align_corners=False)
            x3 = F.interpolate(x[3], size=(height, width), mode='bilinear', align_corners=False)
            x = torch.cat([x[0], x1, x2, x3], 1)
            x = self.head(x)
            return x
        elif self.cfg.MODEL.BACKBONE.HRNET.HEAD_TYPE == "V1":
            return y_list[0]

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def init_pretrained_weights(key):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import errno
    import os
    import wget

    def _get_torch_home():
        ENV_TORCH_HOME = "TORCH_HOME"
        ENV_XDG_CACHE_HOME = "XDG_CACHE_HOME"
        DEFAULT_CACHE_DIR = "~/.cache"
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), "torch"),
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, "checkpoints")
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    filename = f"{key}_imagenet_pretrained.pth"
    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        if comm.is_main_process():
            if model_urls[key] is not None:
                try:
                    wget.download(model_urls[key], cached_file)
                except OSError as e:
                    raise FileNotFoundError(
                        f"Cannot download checkpoint file {filename} from {model_urls[key]}, \nplease check the specified url or manually download it from {model_url}"
                    )
            else:
                raise FileNotFoundError(
                    f"No url specified for {filename},\nplease manually download it from {model_url}"
                )

    comm.synchronize()

    logger.info(f"Loading pretrained model {key} from {cached_file}")
    state_dict = torch.load(cached_file, map_location=torch.device("cpu"))

    return state_dict


@BACKBONE_REGISTRY.register()
def build_hrnet_backbone(cfg):
    """
    Create a HRNet instance from config.
    Returns:
        HRNet: a :class:`HRNet` instance
    """
    # fmt: off
    pretrain = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    # with_ibn      = cfg.MODEL.BACKBONE.WITH_IBN
    # bn_norm       = cfg.MODEL.BACKBONE.NORM
    depth = cfg.MODEL.BACKBONE.DEPTH
    head_type = cfg.MODEL.HRNET.HEAD_TYPE
    # fmt: on

    num_modules_per_stage = {
        "w18": [1, 1, 4, 3],
        "w32": [1, 1, 4, 3],
    }[depth]

    num_branches_per_stage = {
        "w18": [1, 2, 3, 4],
        "w32": [1, 2, 3, 4],
    }[depth]

    type_block_per_stage = {
        "w18": ["BOTTLENECK", "BASIC", "BASIC", "BASIC"],
        "w32": ["BOTTLENECK", "BASIC", "BASIC", "BASIC"],
    }[depth]

    num_blocks_per_stage = {
        "w18": [(4,), (4, 4), (4, 4, 4), (4, 4, 4, 4)],
        "w32": [(4,), (4, 4), (4, 4, 4), (4, 4, 4, 4)],
    }[depth]

    num_channels_per_stage = {
        "w18": [(64,), (18, 36), (18, 36, 72), (18, 36, 72, 144)],
        "w32": [(64,), (32, 64), (32, 64, 128), (32, 64, 128, 256)],
    }[depth]

    fuse_method_per_stage = {
        "w18": ["SUM", "SUM", "SUM", "SUM"],
        "w32": ["SUM", "SUM", "SUM", "SUM"],
    }[depth]

    stage_configs = [
        {
            "NUM_MODULES": num_modules_per_stage[i],
            "NUM_BRANCHES": num_branches_per_stage[i],
            "BLOCK": type_block_per_stage[i],
            "NUM_BLOCKS": num_blocks_per_stage[i],
            "NUM_CHANNELS": num_channels_per_stage[i],
            "FUSE_METHOD": fuse_method_per_stage[i],
        }
        for i in range(4)
    ]

    model = HRNet(cfg, stage_configs)

    if pretrain:
        # Load pretrain path if specifically
        if pretrain_path:
            try:
                state_dict = torch.load(pretrain_path, map_location=torch.device("cpu"))
                logger.info(f"Loading pretrained model from {pretrain_path}")
            except FileNotFoundError as e:
                logger.info(f"{pretrain_path} is not found! Please check this path.")
                raise e
            except KeyError as e:
                logger.info("State dict keys error! Please check the state dict.")
                raise e
        else:
            pretrain_key = f"hrnetv2_{depth}"
            state_dict = init_pretrained_weights(pretrain_key)

        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(get_missing_parameters_message(incompatible.missing_keys))
        if incompatible.unexpected_keys:
            logger.info(get_unexpected_parameters_message(incompatible.unexpected_keys))
    return model


# if __name__ == "__main__":
#     model = build_hrnet_backbone(None)
#     # print(model)

#     state_dict = init_pretrained_weights("hrnetv2_w32")

#     incompatible = model.load_state_dict(state_dict, strict=False)

#     print(incompatible)
