from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import math
import torch.multiprocessing


__all__ = ["ArcFaceLM","ArcFaceLM_DK"]

class ArcFaceLM(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """
    def __init__(self, easy_margin=False, adacos=False,warm_batch = 2000,s = 64,m = 0.5):
        super(ArcFaceLM, self).__init__()
        # s = 64
        # m = 0.5
        self.LOSS = nn.CrossEntropyLoss()
        self.adacos = adacos
        self.s = s
        if self.adacos:
            self.decay_steps = 5000
            self.decay_rate = 0.9
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.warm_batch = warm_batch
    def forward(self, inputs,targets=None,batch=None,total_batch=None,):
        cosine = inputs
        ###=========== warmup =========####
        if batch < self.warm_batch:
            return cosine * self.s

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
        if self.adacos and batch != 0:
            theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
            with torch.no_grad():
                # B_avg = torch.where(one_hot < 1, torch.exp(self.s * cosine), torch.zeros_like(cosine))
                B_avg_ = cosine[one_hot != 1]
                B_avg = torch.sum(torch.exp(self.s * B_avg_)) / input.size(0)
                theta_med = torch.median(theta[one_hot == 1])
                theta_sum = torch.sum(theta[one_hot != 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi / 4 * torch.ones_like(theta_med), theta_med))
            if self.s > 32:
                self.s = 32
            # s = self.s * self.decay_rate ** (batch / self.decay_steps)
            print("=" * 60)
            print("s={} theta_med={} theta_sum={} B_avg={}".format(self.s, theta_med, theta_sum, B_avg))
            print("=" * 60)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)
        output *= self.s
        return output

class ArcFaceLM_DK(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """
    def __init__(self, easy_margin=False, adacos=False,warm_batch = 2000):
        super(ArcFaceLM_DK, self).__init__()
        s = 64
        m = 0.5
        self.LOSS = nn.CrossEntropyLoss()
        self.adacos = adacos
        self.s = s
        if self.adacos:
            self.decay_steps = 5000
            self.decay_rate = 0.9
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.warm_batch = warm_batch
    def forward(self, inputs,targets=None,batch=None,total_batch=None,):
        cosine      = inputs[0]
        soft_cosine = inputs[1]


        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cosine).detach()
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)


        Tempture = 0.1
        Theta_soft_cosine = torch.acos(soft_cosine).detach()

        postive_Theta_soft_cosine = (torch.clamp(math.pi/2 - Theta_soft_cosine,min=0)*Tempture).detach()
        # Theta_soft_cosine = (Theta_soft_cosine*Tempture).detach()

        Theta_cosine = torch.acos(cosine)
        Theta_cosine = one_hot * (Theta_cosine )  +   \
                        (1.0 - one_hot) * (Theta_cosine+postive_Theta_soft_cosine)
        cosine       = torch.cos(Theta_cosine)

        # cosine = (one_hot * cosine) + (
        #         (1.0 - one_hot) * phi_cosine)

        ###=========== warmup =========####
        if batch < self.warm_batch:
            return cosine * self.s

        ####======= add Margin ========####
        phi_sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - phi_sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            with torch.no_grad():
                soft_cosine_norm = (soft_cosine-1)*0.5
                th = self.th + soft_cosine_norm
                mm = soft_cosine_norm + self.mm
            phi = torch.where(cosine > th, phi, cosine - mm )

        if self.adacos and batch != 0:
            theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
            with torch.no_grad():
                # B_avg = torch.where(one_hot < 1, torch.exp(self.s * cosine), torch.zeros_like(cosine))
                B_avg_ = cosine[one_hot != 1]
                B_avg = torch.sum(torch.exp(self.s * B_avg_)) / input.size(0)
                theta_med = torch.median(theta[one_hot == 1])
                theta_sum = torch.sum(theta[one_hot != 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi / 4 * torch.ones_like(theta_med), theta_med))
            if self.s > 32:
                self.s = 32
            # s = self.s * self.decay_rate ** (batch / self.decay_steps)
            print("=" * 60)
            print("s={} theta_med={} theta_sum={} B_avg={}".format(self.s, theta_med, theta_sum, B_avg))
            print("=" * 60)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)
        output *= self.s
        return output


LargMargin = {
    'ArcFace': ArcFaceLM,
    'ArcFaceDK':ArcFaceLM_DK,
}

def build_LargeMargin(cfg):
    AddLargMargin = LargMargin[cfg.MODEL.FACE_HEADS.FACE_HEAD]
    return AddLargMargin