from .face_trainer import *
from .distribute_face_trainer import FaceDistributedDataParallel
from ..backbone import build_backbone,build_face_backbone,build_teacher_student_backbone
from ..face_heads import build_face_head,build_dist_face_heads
_FACE_META_ARCHITECTURES = {"GeneralizedFaceNet": face_trainer,
                            "GeneralizedFaceNetTriplet":face_trainer_triplet,
                            }

def build_face_trainer(cfg,NEED_FC = True):
    meta_arch = _FACE_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    backbone = build_backbone(cfg)
    if NEED_FC:
        head = build_face_head(cfg)
    else:
        head = None
    model    = meta_arch(cfg,backbone,head)
    return  model


def build_dist_face_trainer(cfg,local_rank):
    head = build_dist_face_heads(cfg, local_rank)
    backbone = build_face_backbone(cfg)
    return backbone,head

def build_dist_dk_face_trainer(cfg,local_rank):
    head = build_dist_face_heads(cfg, local_rank)
    teacher,student = build_teacher_student_backbone(cfg)
    return teacher,student,head
