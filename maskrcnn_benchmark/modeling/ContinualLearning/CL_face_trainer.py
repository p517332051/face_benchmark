import torch
from torch import nn
from .DircmModule import BUILD_Dircm
#####========  face trainer =========#######
class face_cl_trainer(nn.Module):
    def __init__(self,cfg, FreezeBACKBONE,BACKBONE,HEAD):
        super(face_cl_trainer, self).__init__()
        self.FreezeBACKBONE = FreezeBACKBONE
        self.BACKBONE = BACKBONE
        self.HEAD     = HEAD
        self.Dircm    = BUILD_Dircm()
    def forward(self, inputs,targets=None,batch=None,total_batch=None ):
        features = self.BACKBONE(inputs)
        if self.training:
            with torch.no_grad():
                Freezefeatures = self.FreezeBACKBONE(inputs)

            outputs = self.HEAD(features, targets, batch, total_batch)
            losses = {}
            losses.update(outputs)
            return losses
        return features


#####========  triplet face trainer =========#######
class face_cl_trainer_triplet(nn.Module):
    def __init__(self,cfg, BACKBONE, HEAD):
        super(face_cl_trainer_triplet, self).__init__()
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