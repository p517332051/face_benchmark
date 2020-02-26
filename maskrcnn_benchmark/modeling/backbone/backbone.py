# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet
from .face_backbone import *
from .face_backbone.OctResNet import *
from .face_backbone.EfficientPolyFace import *
from maskrcnn_benchmark.modeling.face_backbone import Dist_face_backbone
@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model

@registry.BACKBONES.register("FACE-NET-IR-50")
def build_FACE_NET_IR_50(cfg):
    model = IR_50(cfg)
    return model
@registry.BACKBONES.register("FACE-NET-IR-SE-101")
def build_FACE_NET_IR_SE_101(cfg):
    model = IR_SE_101(cfg)
    return model
@registry.BACKBONES.register("FACE-NET-IR-101")
def build_FACE_NET_IR_101(cfg):
    model = IR_101(cfg)
    return model
@registry.BACKBONES.register("FACE-NET-IR-152")
def build_FACE_NET_IR_152(cfg):
    model = IR_152(cfg)
    return model


@registry.BACKBONES.register("FACE-NET-IR-VConv-101")
def build_FACE_NET_IR_OCT_50(cfg):
    model = IR_VConv_101(cfg)
    # model = oct_resnet101(cfg)
    return model

@registry.BACKBONES.register("FACE-NET-Y2MoblieNet")
def build_FACE_NET_IR_OCT_50(cfg):
    model = y2(cfg)
    # model = oct_resnet101(cfg)
    return model

@registry.BACKBONES.register("FACE-NET-MoblieNet")
def build_FACE_NET_IR_OCT_50(cfg):
    model = MobileFacenet(cfg)

    return model

@registry.BACKBONES.register("FACE-NET-fficientPolyFace")
def build_FACE_NET_IR_OCT_50(cfg):
    model = apolynet_stodepth_deeper(cfg.BACKBONE.BACKBONE_OUT_CHANNELS)
    return model

@registry.BACKBONES.register("FACE-NET-IR-OCT-101")
def build_FACE_NET_IR_OCT_50(cfg):
    model = oct_resnet101(cfg)
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)


def build_face_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    backbone = build_backbone(cfg)
    face_backbone = Dist_face_backbone(cfg, backbone)
    return face_backbone

def build_teacher_student_backbone(cfg):
    assert cfg.MODEL.BACKBONE.TEACHER_CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.TEACHER_CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.TEACHER_CONV_BODY
        )
    assert cfg.MODEL.BACKBONE.STUDENT_CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.STUDENT_CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.STUDENT_CONV_BODY
        )
    TEACHER = registry.BACKBONES[cfg.MODEL.BACKBONE.TEACHER_CONV_BODY](cfg)
    STUDENT = registry.BACKBONES[cfg.MODEL.BACKBONE.STUDENT_CONV_BODY](cfg)
    TEACHER = Dist_face_backbone(cfg, TEACHER)
    STUDENT = Dist_face_backbone(cfg, STUDENT)
    return TEACHER,STUDENT
