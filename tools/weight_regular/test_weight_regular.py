import torch,os,sys
sys.path.insert(0,"/data/hongwei/face_benchmark")
sys.path.insert(0,"/data/hongwei/face_benchmark/tools/weight_regular")
from torch import nn

import numpy as np
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import math
# tools/weight_regular/utils.py
# from .utils import plot_result
from maskrcnn_benchmark.config import face_cfg as cfg
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects




def plot_result(embeds, labels,output_path,label_nums = 10,):
    # vis, plot code from https://github.com/pangyupo/mxnet_center_loss
    num = len(labels)
    names = dict()
    for i in range(label_nums):
        names[i] = str(i)
    palette = np.array(sns.color_palette("hls", label_nums))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    sc = ax.scatter(embeds[:, 0], embeds[:, 1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])

    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range(label_nums):
        # Position of each label.
        xtext, ytext = np.median(embeds[labels == i, :], axis=0)
        txt = ax.text(xtext, ytext, names[i])
        ax.arrow(0,0, xtext, ytext,
                 width=0.00001,
                 length_includes_head=True,  # 增加的长度包含箭头部分
                 head_width=0.0001,
                 head_length=0.0001,
                 fc='r',
                 ec='b')
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        # txts.append(txt)
    # plt.show()
    plt.savefig(output_path)








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
    def __init__(self, cfg, easy_margin=False, adacos=False,s=64):
        super(ArcFace, self).__init__()
        # s = 64
        m = 0.1

        self.in_features = cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS
        self.out_features = cfg.MODEL.FACE_HEADS.CLASS_NUMS
        # self.LOSS = nn.BCELoss()
        self.softmax = F.softmax
        self.LOSS = nn.CrossEntropyLoss()
        self.Sigmoid = nn.Sigmoid()
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
        self.warm_batch = 7000
    def forward(self, input, label, batch, batch_total=None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        # print("batchn:  ",input.size() )
        def wx_norm(sub_weights, x):
            temp_x = x
            weight = sub_weights
            cosine = F.linear(F.normalize(temp_x), F.normalize(weight))
            return cosine.cuda()
        def wx(sub_weights, x):
            temp_x = x
            weight = sub_weights
            cosine = F.linear(temp_x, weight)
            return cosine.cuda()
        ###=========== warmup =========####
        # if batch<self.warm_batch:
        if True:
            cosine = wx_norm(self.weight, input)
            # out = self.softmax(cosine)
            # cosine = self.Sigmoid(cosine*self.s)   softmax_result.

            cosine = cosine *self.s
            softmax_result = F.softmax(cosine, dim=0)
            #softmax_result.cpu().detach().numpy()
            loss = self.LOSS(cosine, label)
            loss_dict = {"CrossEntropyLoss": loss}
            return loss_dict

def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.
    print('learning rate is {}'.format(params['lr']))
def main():
    cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS = 2
    cfg.MODEL.FACE_HEADS.CLASS_NUMS = 100


    scale = 64
    eye = np.eye(cfg.MODEL.FACE_HEADS.CLASS_NUMS).astype(np.float)*(scale-0.01)
    weights = np.ones(cfg.MODEL.FACE_HEADS.CLASS_NUMS).astype(np.float)*scale
    weights = weights-eye
    weights = torch.Tensor(weights).cuda()
    cArcFace = ArcFace(cfg, s=weights).cuda()
    label_ =  np.array(list(range(cfg.MODEL.FACE_HEADS.CLASS_NUMS)) )
    optimizer = optim.SGD(cArcFace.parameters(), lr=0.01, momentum=0.9)
    label = torch.Tensor(label_).long().cuda()
    init_weight = F.normalize(cArcFace.weight).cpu().detach().numpy()


    for i in range(20000):
        Input = cArcFace.weight.detach()
        loss = cArcFace(Input,label,i)
        optimizer.zero_grad()
        loss["CrossEntropyLoss"].backward()
        if i==20000:
            schedule_lr(optimizer)
        if i==30000:
            schedule_lr(optimizer)

        if i%100==0:
            print("batch:",i," ",loss["CrossEntropyLoss"])
        optimizer.step()

    plot_result(init_weight, label_, os.path.join("arcloss-train-{}.png".format("init_weight")),label_nums=cfg.MODEL.FACE_HEADS.CLASS_NUMS)
    plot_result(F.normalize(cArcFace.weight).cpu().detach().numpy(), label_, os.path.join("arcloss-train-{}.png".format("finanl_weight")),label_nums = cfg.MODEL.FACE_HEADS.CLASS_NUMS)


if __name__ == '__main__':
    main()  #128 1024 1024 128