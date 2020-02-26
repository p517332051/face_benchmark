from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
# Support: ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
__all__ = ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
class Softmax(nn.Module):
    r"""Implement of Softmax (normal classification head):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel. 
                       if device_id=None, it will be trained on CPU without model parallel.
        """

    def __init__(self, in_features, out_features, device_id):
        super(Softmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zero_(self.bias)

    def forward(self, x):
        if self.device_id == None:
            out = F.linear(x, self.weight, self.bias)
        else:
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            sub_biases = torch.chunk(self.bias, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            bias = sub_biases[0].cuda(self.device_id[0])
            out = F.linear(temp_x, weight, bias)
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                bias = sub_biases[i].cuda(self.device_id[i])
                out = torch.cat((out, F.linear(temp_x, weight, bias).cuda(self.device_id[0])), dim=1)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


class ArcFace(nn.Module):
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
    def __init__(self, cfg, easy_margin=False, adacos=False):
        super(ArcFace, self).__init__()
        s = 64
        m = 0.5

        self.in_features = cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS
        self.out_features = cfg.MODEL.FACE_HEADS.CLASS_NUMS
        self.LOSS = nn.CrossEntropyLoss()
        self.adacos = adacos
        self.s = s
        if self.adacos:
            self.decay_steps = 5000
            self.decay_rate = 0.9
        self.m = m
        self.weight = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.warm_batch = 2000
    def forward(self, input, label, batch, batch_total):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # print("batchn:  ",input.size() )
        def wx_norm(sub_weights, x):
            temp_x = x
            weight = sub_weights
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            return cosine.cuda()
        ###=========== warmup =========####
        if batch<self.warm_batch:
            cosine = wx_norm(self.weight, input)
            loss = self.LOSS(cosine * self.s, label)
            loss_dict = {"CrossEntropyLoss": loss}
            return loss_dict

        x = input
        cosine = wx_norm(self.weight, x)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
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
        loss = self.LOSS(output, label)
        loss_dict = {"CrossEntropyLoss": loss}
        return loss_dict


class ExclusiveLinear(nn.Module):
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

    def __init__(self, in_features, out_features, device_id, s=64.0, m=0.50, easy_margin=False):
        super(ExclusiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id

        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            weight_norm = F.normalize(self.weight)
            cos = torch.mm(weight_norm, weight_norm.t())

            cos.clamp(-1, 1)
            cos1 = cos.detach()
            cos1.scatter_(1, torch.arange(self.out_features).view(-1, 1).long().cuda(self.device_id[0]), -100)
            max_cos, indices = torch.max(cos1, dim=1)
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))

            temp_weight = self.weight.cuda(self.device_id[0])
            cos = torch.mm(F.normalize(weight), F.normalize(temp_weight).t())

            cos.clamp(-1, 1)
            cos1 = cos.detach()
            length = weight.size()[0]
            cos1.scatter_(1, torch.arange(length).view(-1, 1).long().cuda(self.device_id[0]), -100)
            max_cos, indices = torch.max(cos1, dim=1)

            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])),
                                   dim=1)

                temp_weight = self.weight.cuda(self.device_id[i])
                # cos = torch.cat((cos, torch.mm(F.normalize(weight_transform), F.normalize(temp_weight).t()).cuda(self.device_id[0])), dim=0)
                cos = torch.mm(F.normalize(weight), F.normalize(temp_weight).t())

                cos.clamp(-1, 1)
                cos1 = cos.detach()
                length = weight.size()[0]
                cos1.scatter_(1, torch.arange(length).view(-1, 1).long().cuda(self.device_id[i]), -100)

                max_cos_, indices = torch.max(cos1, dim=1)
                max_cos = torch.cat((max_cos, max_cos_.cuda(self.device_id[0])), dim=0)

        exclusive_loss = torch.sum(max_cos) / self.out_features

        # cos1.scatter_(1, torch.arange(self.out_features).view(-1, 1).long().cuda(self.device_id[0]), -100)
        # mask = torch.zeros((self.out_features, self.out_features)).cuda(self.device_id[0])
        # mask.scatter_(1, indices.view(-1, 1).long(), 1)
        #
        # exclusive_loss = torch.dot(cos.view(cos.numel()), mask.view(mask.numel())) / self.out_features

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        with torch.no_grad():
            # B_avg = torch.where(one_hot < 1, torch.exp(self.s * cosine), torch.zeros_like(cosine))
            B_avg_ = cosine[one_hot != 1]
            B_avg = torch.sum(torch.exp(self.s * B_avg_)) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            theta_sum = torch.sum(theta[one_hot != 1])

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4

        print("=" * 60)
        print("s={} theta_med={} theta_sum={} B_avg={}".format(self.s, theta_med, theta_sum, B_avg))
        print("=" * 60)

        output *= self.s

        return output, exclusive_loss


# class AdaCos(nn.Module):
#     def __init__(self, num_features, num_classes, m=0.50):
#         super(AdaCos, self).__init__()
#         self.num_features = num_features
#         self.n_classes = num_classes
#
#         self.s = math.sqrt(2) * math.log(num_classes - 1)
#         self.m = m
#
#         self.w = Parameter(torch.FloatTensor(num_classes, num_features))
#         nn.init.xavier_uniform_(self.w)
#
#     def forward(self, input, label=None):
#         # normalize features
#         x = F.normalize(input)
#         # normalize weights
#         w = F.normalize(self.w)
#         # dot product
#         logits = F.linear(x, w)
#         if label is None:
#             return logits
#         # feature re-scale
#         theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
#         one_hot = torch.zeros_like(logits)
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         with torch.no_grad():
#             B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
#             B_avg = torch.sum(B_avg) / input.size(0)
#             # print(B_avg)
#             theta_med = torch.median(theta[one_hot == 1])
#             self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
#         output = self.s * logits
#
#         return output


# class ExclusiveLinear(nn.Module):
#     def __init__(self, feat_dim, num_class):
#         super(ExclusiveLinear, self).__init__()
#         self.num_class = num_class
#         self.feat_dim = feat_dim
#
#         self.weight = nn.Parameter(torch.randn(self.num_class, self.feat_dim))
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#
#     def forward(self):
#         weight_norm = F.normalize(self.weight, p=2, dim=1)
#         cos = torch.mm(weight_norm, weight_norm.t())
#         cos.clamp(-1, 1)
#
#         cos1 = cos.detach()
#         cos1.scatter_(1, torch.arange(self.num_class).view(-1, 1).long().cuda(), -100)
#
#         _, indices = torch.max(cos1, dim=0)
#         mask = torch.zeros((self.num_class, self.num_class)).cuda()
#         mask.scatter_(1, indices.view(-1, 1).long(), 1)
#
#         exclusive_loss = torch.dot(cos.view(cos.numel()), mask.view(mask.numel())) / self.num_class
#
#         return exclusive_loss


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes, feat_dim=512, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


class CosFace(nn.Module):
    r"""Implement of CosFace (https://arxiv.org/pdf/1801.09414.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
        s: norm of input feature
        m: margin
        cos(theta)-m
    """

    def __init__(self, in_features, out_features, device_id, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_id = device_id
        self.s = s
        self.m = m

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cosine = torch.cat((cosine, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])),
                                   dim=1)
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[1])
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                    (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', s = ' + str(self.s) \
               + ', m = ' + str(self.m) + ')'


class SphereFace(nn.Module):
    r"""Implement of SphereFace (https://arxiv.org/pdf/1704.08063.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
        m: margin
        cos(m*theta)
    """

    def __init__(self, in_features, out_features, device_id, m=4):
        super(SphereFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.device_id = device_id

        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        if self.device_id == None:
            cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        else:
            x = input
            sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            temp_x = x.cuda(self.device_id[0])
            weight = sub_weights[0].cuda(self.device_id[0])
            cos_theta = F.linear(F.normalize(temp_x), F.normalize(weight))
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                weight = sub_weights[i].cuda(self.device_id[i])
                cos_theta = torch.cat(
                    (cos_theta, F.linear(F.normalize(temp_x), F.normalize(weight)).cuda(self.device_id[0])), dim=1)

        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        if self.device_id != None:
            one_hot = one_hot.cuda(self.device_id[0])
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features = ' + str(self.in_features) \
               + ', out_features = ' + str(self.out_features) \
               + ', m = ' + str(self.m) + ')'


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class Am_softmax(nn.Module):
    r"""Implement of Am_softmax (https://arxiv.org/pdf/1801.05599.pdf):
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
        m: margin
        s: scale of outputs
    """

    def __init__(self, in_features, out_features, device_id, m=0.35, s=30.0):
        super(Am_softmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.device_id = device_id

        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)  # initialize kernel

    def forward(self, embbedings, label):
        if self.device_id == None:
            kernel_norm = l2_norm(self.kernel, axis=0)
            cos_theta = torch.mm(embbedings, kernel_norm)
        else:
            x = embbedings
            sub_kernels = torch.chunk(self.kernel, len(self.device_id), dim=1)
            temp_x = x.cuda(self.device_id[0])
            kernel_norm = l2_norm(sub_kernels[0], axis=0).cuda(self.device_id[0])
            cos_theta = torch.mm(temp_x, kernel_norm)
            for i in range(1, len(self.device_id)):
                temp_x = x.cuda(self.device_id[i])
                kernel_norm = l2_norm(sub_kernels[i], axis=0).cuda(self.device_id[i])
                cos_theta = torch.cat((cos_theta, torch.mm(temp_x, kernel_norm).cuda(self.device_id[0])), dim=1)

        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        phi = cos_theta - self.m
        label = label.view(-1, 1)  # size=(B,1)
        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, label.data.view(-1, 1), 1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = phi[index]  # only change the correct predicted output
        output *= self.s  # scale up in order to make softmax work, first introduced in normface

        return output
