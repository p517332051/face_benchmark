# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN,GeneralizedRCNN_G,GeneralizedFaceNet

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN,
                                 "GeneralizedRCNN_G":GeneralizedRCNN_G,
                                 "GeneralizedFaceNet_Head": GeneralizedFaceNet,
                                 }
__all__ = ["build_detection_model","build_backbone_model","build_head_model"]

def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)

def build_backbone_model(cfg):
    meta_arch = GeneralizedRCNN_G
    return meta_arch(cfg)

def build_head_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)

