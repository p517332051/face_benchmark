# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .lr_scheduler import WarmupMultiStepLR
from maskrcnn_benchmark.solver.diffGrad import diffGrad

def make_optimizer_diffGrad(cfg, model):
    nb_iters = 300
    lrn_rate = 0.1
    beta1 = 0.95
    beta2 = 0.999
    eps = 0.00000001
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

        if "BACKBONE.output_layer.4.weight" in key:
            weight_decay *= 10
        if "BACKBONE.linear.*.weight" in key:
            weight_decay *= 10
        params += [{"params": [value], "betas": (beta1,beta2),"eps":eps}]


    optimizer = diffGrad(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer



def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

        # if "BACKBONE.output_layer.4.weight" in key:
        #     weight_decay *= 10
        # if "BACKBONE.linear.*.weight" in key:
        #     weight_decay *= 10
        if "running_mean" in key:
            weight_decay = 0
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS,
        cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
    )
