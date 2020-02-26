from .CL_face_trainer import *
from ..backbone import build_backbone
from ..face_heads import build_face_head
_FACE_CL_META_ARCHITECTURES = {"GeneralizedCLFaceNet": face_cl_trainer,
                            "GeneralizedCLFaceNetTriplet":face_cl_trainer_triplet,
                            }
def build_face_trainer(cfg):
    meta_arch = _FACE_CL_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    backbone = build_backbone(cfg)
    head     = build_face_head(cfg)
    return meta_arch(cfg,backbone,head)