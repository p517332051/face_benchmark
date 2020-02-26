from .metrics import *
from maskrcnn_benchmark.utils.comm import get_world_size
face_heads = {
    'ArcFace': ArcFace,
    'CosFace': CosFace,
    'SphereFace': SphereFace,
    'Am_softmax': Am_softmax,
}


def build_face_head(cfg):
    head = face_heads[cfg.MODEL.RPN.RPN_HEAD]

    return head(in_features=cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS,out_features = cfg.MODEL.RPN.CLASS_NUMS,device_id=get_world_size(),s=64)


#in_features=EMBEDDING_SIZE, out_features=NUM_CLASS, device_id=GPU_ID, s=32
