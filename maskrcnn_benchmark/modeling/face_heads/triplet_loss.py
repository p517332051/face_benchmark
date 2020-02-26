import torch
import numpy as np
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import math
from .metrics import ArcFace

class AngularlDistance(Function):
    def __init__(self,):
        super(AngularlDistance, self).__init__()
    def __call__(self, x1, x2):
        assert x1.size() == x2.size()
        cosine = F.linear(F.normalize(x1), F.normalize(x2))
        return (1-cosine)/2.0
class PairwiseDistance(Function):
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def __call__(self, x1, x2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)
class TripletMarginLoss(Function):
    """Triplet loss function.
    """
    def __init__(self, margin=0.3,):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        # self.pdist = PairwiseDistance(2)  # norm 2
        self.pdist = AngularlDistance()
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)
    def __call__(self, anchor, positive, negative,batch=None):
        d_p = self.pdist(anchor, positive)*64
        d_n = self.pdist(anchor, negative)*64
        y = torch.ones_like(d_n)*self.margin*64
        loss = self.ranking_loss(d_n, d_p, y)
        triplet_loss = {"TripletLoss": loss}
        return triplet_loss

# class TripletLoss(object):
#     """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
#     Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
#     Loss for Person Re-Identification'."""
#
#     def __init__(self, margin=None):
#         self.margin = margin
#         if margin is not None:
#             self.ranking_loss = nn.MarginRankingLoss(margin=margin)
#         else:
#             self.ranking_loss = nn.SoftMarginLoss()
#
#     def __call__(self, global_feat, labels, normalize_feature=False):
#         if normalize_feature:
#             global_feat = normalize(global_feat, axis=-1)
#         dist_mat = euclidean_dist(global_feat, global_feat)
#         dist_ap, dist_an = hard_example_mining(
#             dist_mat, labels)
#         y = dist_an.new().resize_as_(dist_an).fill_(1)
#         if self.margin is not None:
#             loss = self.ranking_loss(dist_an, dist_ap, y)
#         else:
#             loss = self.ranking_loss(dist_an - dist_ap, y)
#         return loss, dist_ap, dist_an



# class Addmargin(Function):
#     def __init__(self,margin):
#         super(Addmargin, self).__init__()
#         if margin < 0:
#             flag = -1
#         else:
#             flag = 1
#         self.th = math.cos(math.pi - margin)
#         self.mm = math.sin(math.pi - margin) * margin
#         self.cos_m = flag * math.cos(margin)
#         self.sin_m = math.sin(margin)
#     def __call__(self,cosine):
#         sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
#         phi = cosine * self.cos_m - sine * self.sin_m
#         # phi = torch.where(cosine > 0, phi, cosine)
#         phi = torch.where(cosine > self.th, phi, cosine - self.mm)
#         return phi.cuda()


# def Addmargin(cosine,margin):
#     if margin<0:
#         flag=-1
#     else:
#         flag=1
#     th    = math.cos(math.pi - margin)
#     mm    = math.sin(math.pi - margin) * margin
#     cos_m = flag*math.cos(margin)
#     sin_m = math.sin(margin)
#     sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
#     phi = cosine * cos_m - sine * sin_m
#     phi = torch.where(cosine > th, phi, cosine - mm)
    # return phi

# class TripletMarginLoss(Function):
#     """Triplet loss function.
#     """
#     def __init__(self, margin=0,s=16):
#         super(TripletMarginLoss, self).__init__()
#         self.margin = margin
#         self.scale  = s
#         self.warmup = 0
#         self.ComputeDistance = AngularlDistance()
#
#         self.LOSS = nn.BCELoss()
#         self.Addmargin = Addmargin(margin)
#     def __call__(self, anchor, positive, negative,batch):
#         d_p = self.ComputeDistance(positive,anchor)
#         d_n = self.ComputeDistance(negative,anchor )
#
#         # if batch>self.warmup:
#         # d_p = self.Addmargin(d_p, )
#         # d_n = self.Addmargin(d_n,)
#         logitc = -self.margin+ d_p - d_n
#         label = torch.ones_like(d_p).float().cuda()
#         label[label != 1.0] = 1.0
#         loss = self.LOSS(torch.sigmoid(self.scale*(logitc)),label.detach())
#         triplet_loss = {"CrossEntropyLoss_TripletLoss": loss}
#         return triplet_loss
class triplet_loss(nn.Module):
    def __init__(self, cfg):
        super(triplet_loss, self).__init__()
        self.TripletMarginLoss = TripletMarginLoss(cfg.SOLVER.TRIPLET_MARGIN)
    def forward(self, input, label, batch, total_batch):
        losses = {}
        out_a, out_p,out_n = input
        label_p,label_n = label
        ############===========    TripletLoss   =============############
        Triplet_loss = self.TripletMarginLoss(out_a, out_p, out_n,batch)
        losses.update(Triplet_loss)
        return losses
