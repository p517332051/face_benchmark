from .metrics import *
from .triplet_loss import  *
from .Dist_Face_Head import Dist_Face_Head
face_heads = {
    'ArcFace': ArcFace,
    'CosFace': CosFace,
    'SphereFace': SphereFace,
    'Am_softmax': Am_softmax,
    'triplet':triplet_loss,
}

def build_face_head(cfg):
    head = face_heads[cfg.MODEL.FACE_HEADS.FACE_HEAD]
    return head(cfg)
def build_dist_face_heads(cfg,local_rank):
    head = Dist_Face_Head(cfg,GPUS=local_rank)
    return head
