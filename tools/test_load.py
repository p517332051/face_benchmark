import torch


model_root = '/data/hongwei/FCOS/BACKBONE_0000000.pth'
BACKBONE = torch.load(model_root, )

print(BACKBONE)