from __future__ import print_function
from __future__ import division
import torch

import torch.nn as nn
import torch.nn.functional as F
import math
import threading
import torch.multiprocessing
from torch.nn import Parameter
from torch._utils import ExceptionWrapper
from maskrcnn_benchmark.utils.comm import get_world_size
from .AddLargeMarigin import build_LargeMargin

class Dist_FC(nn.Module):
    def __init__(self, cfg,GPUS = 0):
        super(Dist_FC, self).__init__()
        self.world_size = get_world_size()
        self.GPUS = GPUS
        self.in_features = cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS
        self.out_features = cfg.MODEL.FACE_HEADS.CLASS_NUMS

        weights = torch.FloatTensor(self.out_features, self.in_features)
        # torch.manual_seed(42)
        nn.init.xavier_uniform_(weights)
        weights = torch.chunk(weights, len(GPUS), dim=0)
        self.weights = nn.ParameterList([nn.Parameter(weight.to(GPU)) for weight,GPU in zip(weights,self.GPUS)])

    def forward(self, inputs):
        results = {}
        lock = threading.Lock()
        def wx_norm(sub_weights, temp_x,GPU):
            try:
                cosine = F.linear(F.normalize(temp_x), F.normalize(sub_weights))
                # cosine = F.linear(F.normalize(temp_x), sub_weights)
                cosine = torch.chunk(cosine,len(self.GPUS))
                with lock:
                    results[GPU] = cosine
            except Exception:
                with lock:
                    results[GPU] = ExceptionWrapper(
                        where="weight{GPU}  is error".format(GPU))
        if len(self.weights)>1:
                threads = [threading.Thread(target=wx_norm,
                                            args=(weight, input,GPU))
                           for weight, GPU, input in
                           zip(self.weights, self.GPUS, inputs)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
        else:
            for weight,GPU,input in zip(self.weights,self.GPUS,inputs):
                wx_norm(weight, input, GPU)
        outputs = [results[i] for i in self.GPUS]
        outputs = [torch.cat([output_.to(self.GPUS[i]) for output_ in output], dim=1) for i,output in  enumerate(zip(*outputs))]
        return outputs


class Dist_Face_head_(nn.Module):
    def __init__(self, cfg, easy_margin=False, adacos=False,GPUS = 0):
        super(Dist_Face_head_, self).__init__()
        self.world_size = get_world_size()
        self.GPUS = GPUS
        self.in_features = cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS
        self.out_features = cfg.MODEL.FACE_HEADS.CLASS_NUMS

        weights = torch.FloatTensor(self.out_features, self.in_features)
        # torch.manual_seed(42)
        nn.init.xavier_uniform_(weights)
        weights = torch.chunk(weights, len(GPUS), dim=0)
        self.weights = nn.ParameterList([nn.Parameter(weight.to(GPU)) for weight,GPU in zip(weights,self.GPUS)])
        self.LM = build_LargeMargin(cfg)(easy_margin=easy_margin, adacos=adacos)
        # self.CrossEntropyLossList = nn.Sequential([nn.CrossEntropyLoss() for i in range(len(GPUS))])
        self.CrossEntropyLoss = nn.CrossEntropyLoss()


    def forward(self, inputs,targets=None,soft_target=None,batch=None,total_batch=None ):
        results = {}
        lock = threading.Lock()
        def wx_norm(sub_weights, temp_x,GPU):
            try:
                cosine = F.linear(F.normalize(temp_x), F.normalize(sub_weights))
                # cosine = F.linear(F.normalize(temp_x), sub_weights)
                cosine = torch.chunk(cosine,len(self.GPUS))
                with lock:
                    results[GPU] = cosine
            except Exception:
                with lock:
                    results[GPU] = ExceptionWrapper(
                        where="weight{GPU}  is error".format(GPU))
        if len(self.weights)>1:
                threads = [threading.Thread(target=wx_norm,
                                            args=(weight, input,GPU))
                           for weight, GPU, input in
                           zip(self.weights, self.GPUS, inputs)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
        else:
            for weight,GPU,input in zip(self.weights,self.GPUS,inputs):
                wx_norm(weight, input, GPU)

        #####===== compute LOSSES =====########
        LOSSES = {}
        LOSSES["CrossEntropyLoss"] = 0
        Cosine_MutiGPU = {}
        outputs = [results[i] for i in self.GPUS]
        targets = torch.chunk(targets,len(self.GPUS))
        targets = [target.to(GPU) for target,GPU in zip(targets,self.GPUS)]
        def _work_loss(GPU,inputs, targets,batch,total_batch):
            try:
                inputs = torch.cat([input.to(GPU) for input  in inputs],dim=1)
                cosine   = self.LM(inputs,targets=targets,batch=batch,total_batch=total_batch )
                with lock:
                    Cosine_MutiGPU[GPU] = cosine
            except:
                ExceptionWrapper(
                    where="LOSSES{GPU}  is error".format(GPU))
        if len(self.weights)>1:
        # if False:
                threads = [threading.Thread(target=_work_loss,
                                            args=(self.GPUS[i],output, targets[i],batch,total_batch))
                           for i,output in
                           enumerate(zip(*outputs))]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                for GPU, target in zip(self.GPUS, targets):
                    LOSS = self.CrossEntropyLoss(Cosine_MutiGPU[GPU], target) / len(self.GPUS)
                    LOSSES["CrossEntropyLoss"] += LOSS.cuda()
        else:
            for i,output in enumerate(zip(*outputs)):
                output = torch.cat([output.to(self.GPUS[i]) for output  in output],dim=1)
                cosine   = self.LM(output,targets=targets[i],batch=batch,total_batch=total_batch ).cuda()
                LOSS = self.CrossEntropyLoss(cosine, targets[i]) / len(self.GPUS)
                LOSSES["CrossEntropyLoss"]+=LOSS/len(self.GPUS)
        return LOSSES

class Dist_Face_Head(nn.Module):
    def __init__(self, cfg, easy_margin=True, adacos=False,GPUS = 0):
        super(Dist_Face_Head, self).__init__()
        self.GPUS = GPUS
        self.FC = Dist_FC(cfg,GPUS)
        AddLargeMargin = build_LargeMargin(cfg)
        self.LM = AddLargeMargin(easy_margin=easy_margin, adacos=adacos)

        self.LMDisting = AddLargeMargin(easy_margin=easy_margin, adacos=adacos,  s = 64,m = 0.9)
        self.CE = nn.CrossEntropyLoss()

        self.MSE = nn.MSELoss()
    def forward(self, inputs,targets=None,soft_target=None,batch=None,total_batch=None ):

        outputs = self.FC(inputs=inputs )
        # if soft_target != None:
        #     with torch.no_grad():
        #         soft_targets = self.FC(inputs=soft_target)
        #         soft_targets = [soft_target.detach() for soft_target in soft_targets]
        #     outputs = [[output, soft_target] for  output, soft_target in zip(outputs, soft_targets)]
        #####===== compute LOSSES =====########
        Cosine_MutiGPU = {}
        targets_chunck = torch.chunk(targets,len(self.GPUS))
        targets = {}
        for target, GPU in zip(targets_chunck, self.GPUS):
            targets[GPU] = target.to(GPU)
        if len(self.GPUS)>1:
            lock = threading.Lock()
            def _work_loss(GPU, inputs, targets, batch, total_batch):
                try:
                    cosine = self.LM(inputs, targets=targets, batch=batch, total_batch=total_batch)
                    with lock:
                        Cosine_MutiGPU[GPU] = cosine
                except:
                    ExceptionWrapper(
                        where="LOSSES{GPU}  is error".format(GPU))
            threads = [threading.Thread(target=_work_loss,
                                        args=(GPU, output, targets[GPU], batch, total_batch))
                       for GPU, output in zip(self.GPUS, outputs)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        else:
            for GPU, output in zip(self.GPUS, outputs):
                cosine = self.LM(output, targets=targets[GPU], batch=batch, total_batch=total_batch).cuda()
                Cosine_MutiGPU[GPU] = cosine
        LOSSES = [self.CE(Cosine_MutiGPU[GPU], targets[GPU]).cuda()
                  for GPU in self.GPUS]
        LOSSES_DICT = {}
        LOSSES_DICT["CrossEntropyLoss"] = sum(LOSSES) / len(self.GPUS)


        if soft_target!=None:
            ###强行和 teacher的特征角度拟合
            m = 0.5
            s = 8
            Disting_Output = torch.sum(F.normalize(inputs[0])*F.normalize(soft_target[0].detach()),1)
            Disting_Output = (Disting_Output + 1) / 2
            Disting_Output = torch.cos((torch.acos(Disting_Output)+m ) ) * s
            label = torch.ones_like(Disting_Output) * s
            LOSSES_DICT["DistingLoss"]  = self.MSE(Disting_Output,label)


            ###softmax 相关性矩阵 蒸馏
            # Disting_Output = torch.matmul(F.normalize(inputs[0]),F.normalize(soft_target[0].detach() ).t())
            # with torch.no_grad():
            #     Disting_label = torch.cat([targets[GPU].cuda() for GPU in self.GPUS])
            #     Disting_label = torch.cat([torch.eq(label_, Disting_label).reshape((-1,1)).long() for label_ in Disting_label],dim=1)
            # # Disting_label = torch.range(0, Disting_Output.shape[0])[0:-1].long().cuda()
            # Disting_cosine = self.LMDisting(Disting_Output, targets=Disting_label, batch=batch, total_batch=total_batch)
            # LOSSES_DICT["Disting_CE"] = self.CE(Disting_cosine,Disting_label)
        return LOSSES_DICT


# 1.module.Dist_FC.weights.0
# 1.module.Dist_FC.weights.1
# 1.module.Dist_FC.weights.2
# 1.module.Dist_FC.weights.3