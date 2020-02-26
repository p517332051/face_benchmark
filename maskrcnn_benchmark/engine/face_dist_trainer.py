# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.comm import get_world_size, synchronize
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.utils.divs_images import divs_tensors
from apex import amp


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0,)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def do_face_train_dist_DIV_FC(
        cfg,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        test_period,
        arguments,
        divs_nums,
):
    # model,       head                 = model
    # optimizer,   head_optimizer       = optimizer
    # scheduler,   head_scheduler       = scheduler
    # checkpointer,head_checkpointer    = checkpointer
    head  = model[1]
    model = model[0]
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    dataset_names = cfg.DATASETS.TEST

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration
        images_list,targets_list = divs_tensors(device=device, tensors=images, targets=targets, divs_nums=divs_nums)
        ####======== 拆分batch 可能对bn层有影响 ==========####
        optimizer.zero_grad()
        if len(images_list)>1:
            grad_sync     = False
        else:
            grad_sync     = True
        for i,(images,targets) in enumerate(zip(images_list,targets_list) ):
            # for i in range(100000000):model(inputs = images, repeat_forward=repeat_forward)
            # features  = model(inputs = images,repeat_forward=repeat_forward)
            # features = [feature.detach() for feature in features]
            loss_dict = head(model(inputs = images,grad_sync  = grad_sync),targets = targets,batch=iteration,total_batch=None,grad_sync = grad_sync)#param_sync = param_sync
            losses = sum(loss for loss in loss_dict.values())
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(torch.mean(loss) for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)
            losses/=divs_nums
            with amp.scale_loss(losses, optimizer) as scaled_losses:
                scaled_losses.backward()
            if i==len(images_list)-2:
                grad_sync = True
        optimizer.step()
        scheduler.step()
        # head_optimizer.step()

        # head_scheduler.step()
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            checkpointer.save_backbone("BACKBONE_{:07d}".format(iteration))
            # head_checkpointer.save("HEAD_{:07d}".format(iteration), **arguments)
            # if iteration > 40000:
            #     try:
            #         checkpointer.save_backbone("BACKBONE_{:07d}".format(iteration))
            #     except:
            #         logger.info("save"," BACKBONE_{:07d}".format(iteration)," failed")
        #####========= data test ============#######
        if data_loader_val is not None and test_period > 0 and iteration % test_period == 0:
            meters_val = MetricLogger(delimiter="  ")
            synchronize()
            _ = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                model,
                # The method changes the segmentation mask format in a data loader,
                # so every time a new data loader is created:
                make_data_loader(cfg, is_train=False, is_distributed=(get_world_size() > 1), is_for_period=True),
                dataset_name="[Validation]",
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=None,
            )
            synchronize()
            model.train()
            with torch.no_grad():
                # Should be one image for each GPU:
                for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                    images_val = images_val.to(device)
                    targets_val = [target.to(device) for target in targets_val]
                    loss_dict = model(images_val, targets_val)
                    losses = sum(loss for loss in loss_dict.values())
                    loss_dict_reduced = reduce_loss_dict(loss_dict)
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    meters_val.update(loss=losses_reduced, **loss_dict_reduced)
            synchronize()
            logger.info(
                meters_val.delimiter.join(
                    [
                        "[Validation]: ",
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters_val),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )


        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
            checkpointer.save_backbone("model_final")
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )



def do_face_train_dk_dist_DIV_FC(
        cfg,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        test_period,
        arguments,
        divs_nums,
):
    # model,       head                 = model
    optimizer,   head_optimizer       = optimizer
    scheduler,   head_scheduler       = scheduler
    checkpointer,head_checkpointer    = checkpointer
    teacher = model[2]
    head    = model[1]
    model   = model[0]
    logger  = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters     = MetricLogger(delimiter="  ")
    max_iter   = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    dataset_names = cfg.DATASETS.TEST
    teacher.eval()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration
        images_list,targets_list = divs_tensors(device=device, tensors=images, targets=targets, divs_nums=divs_nums)
        ####======== 拆分batch 可能对bn层有影响 ==========####
        optimizer.zero_grad()
        if len(images_list)>1:
            grad_sync     = False
        else:
            grad_sync     = True
        for i,(images,targets) in enumerate(zip(images_list,targets_list) ):

            with torch.no_grad():
                soft_target = teacher(inputs=images,)# grad_sync=False,grad_params=False)
                # soft_target = [soft_target_.detach() for soft_target_ in soft_target]
                soft_target=[soft_target.to(GPU).detach() for GPU in head.module.GPUS]
            features    = model(inputs=images, grad_sync=grad_sync)
            loss_dict   = head(features,targets = targets,batch=iteration,soft_target=soft_target,total_batch=None,grad_sync = grad_sync)#param_sync = param_sync
            losses = sum(loss for loss in loss_dict.values())
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(torch.mean(loss) for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)
            losses/=divs_nums
            with amp.scale_loss(losses, optimizer) as scaled_losses:
                scaled_losses.backward()
            if i==len(images_list)-2:
                grad_sync = True
        optimizer.step()
        scheduler.step()
        # head_optimizer.step()
        # head_scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            checkpointer.save_backbone("BACKBONE_{:07d}".format(iteration))
            head_checkpointer.save("HEAD_{:07d}".format(iteration), **arguments)
            # if iteration > 40000:
            #     try:
            #         checkpointer.save_backbone("BACKBONE_{:07d}".format(iteration))
            #     except:
            #         logger.info("save"," BACKBONE_{:07d}".format(iteration)," failed")

def do_face_train_dist(
        cfg,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        test_period,
        arguments,
        divs_nums,
):
    model,       head                 = model
    optimizer,   head_optimizer       = optimizer
    scheduler,   head_scheduler       = scheduler
    checkpointer,head_checkpointer    = checkpointer

    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    dataset_names = cfg.DATASETS.TEST
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration
        images_list,targets_list = divs_tensors(device=device, tensors=images, targets=targets, divs_nums=divs_nums)
        ####======== 拆分batch 可能对bn层有影响 ==========####
        optimizer.zero_grad()
        head_optimizer.zero_grad()
        for images,targets in zip(images_list,targets_list):
            features  = model(inputs = images, targets = targets,batch=iteration,total_batch=None)
            loss_dict = head(features,targets = targets,batch=iteration,total_batch=None)
            losses = sum(loss for loss in loss_dict.values())
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(torch.mean(loss) for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)
            losses/=divs_nums
            with amp.scale_loss(losses, optimizer) as scaled_losses:
                scaled_losses.backward()
        optimizer.step()

        scheduler.step()

        # head_optimizer.step()
        # head_scheduler.step()
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("BACKBONE_{:07d}".format(iteration), **arguments)
            head_checkpointer.save("HEAD_{:07d}".format(iteration), **arguments)
            # if iteration > 40000:
            #     try:
            #         checkpointer.save_backbone("BACKBONE_{:07d}".format(iteration))
            #     except:
            #         logger.info("save"," BACKBONE_{:07d}".format(iteration)," failed")
        #####========= data test ============#######
        if data_loader_val is not None and test_period > 0 and iteration % test_period == 0:
            meters_val = MetricLogger(delimiter="  ")
            synchronize()
            _ = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                model,
                # The method changes the segmentation mask format in a data loader,
                # so every time a new data loader is created:
                make_data_loader(cfg, is_train=False, is_distributed=(get_world_size() > 1), is_for_period=True),
                dataset_name="[Validation]",
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=None,
            )
            synchronize()
            model.train()
            with torch.no_grad():
                # Should be one image for each GPU:
                for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                    images_val = images_val.to(device)
                    targets_val = [target.to(device) for target in targets_val]
                    loss_dict = model(images_val, targets_val)
                    losses = sum(loss for loss in loss_dict.values())
                    loss_dict_reduced = reduce_loss_dict(loss_dict)
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    meters_val.update(loss=losses_reduced, **loss_dict_reduced)
            synchronize()
            logger.info(
                meters_val.delimiter.join(
                    [
                        "[Validation]: ",
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters_val),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )


        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
            checkpointer.save_backbone("model_final")
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
