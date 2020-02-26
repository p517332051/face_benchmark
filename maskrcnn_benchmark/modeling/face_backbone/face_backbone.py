import torch
import torch.nn as nn
class Dist_face_backbone(nn.Module):
    def __init__(self,cfg, BACKBONE):
        super(Dist_face_backbone, self).__init__()
        self.BACKBONE = BACKBONE
    def forward(self, inputs):
        features = self.BACKBONE(inputs)
        # features = F.normalize(features)
        return features