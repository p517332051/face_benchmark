import torch
from torch import nn
import torch.nn.functional as F
#####========  face trainer =========#######
class face_trainer(nn.Module):
    def __init__(self,cfg, BACKBONE,HEAD):
        super(face_trainer, self).__init__()
        self.BACKBONE = BACKBONE
        self.HEAD     = HEAD
    def forward(self, inputs,targets=None,batch=None,total_batch=None ):
        features = self.BACKBONE(inputs)
        #{p.device for p in module.parameters()}
        if self.training:
            outputs = self.HEAD(features, targets, batch, total_batch)
            losses = {}
            losses.update(outputs)
            return losses
        return features

#####========  triplet face trainer =========#######
class face_trainer_triplet(nn.Module):
    def __init__(self,cfg, BACKBONE, HEAD):
        super(face_trainer_triplet, self).__init__()
        self.BACKBONE = BACKBONE
        self.HEAD = HEAD
    def forward(self, tensors, targets=None, batch=None, total_batch=None):
        img_a,img_p,img_n = tensors
        label_p,label_n   = targets
        out_a, out_p, out_n = self.BACKBONE(img_a), self.BACKBONE(img_p), self.BACKBONE(img_n)
        outputs = self.HEAD([out_a, out_p, out_n], [label_p,label_n], batch, total_batch)
        losses = {}
        losses.update(outputs)
        return losses