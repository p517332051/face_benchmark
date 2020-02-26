# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference,inference_G_HEAD
from maskrcnn_benchmark.modeling.detector import build_detection_model,build_detection_model_GAN_Disting
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--student-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--teacher-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--teacher-pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters. You can specify parameter file name.')
    parser.add_argument('--student-pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters. You can specify parameter file name.')


    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()



    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())
    #########===================student net=================#############
    cfg.merge_from_file(args.student_file)
    cfg.merge_from_list(args.opts)
    output_dir = cfg.OUTPUT_DIR
    Student_G, _, _ = build_detection_model_GAN_Disting(cfg)
    Student_G.to(cfg.MODEL.DEVICE)
    Student_G_checkpointer = DetectronCheckpointer(cfg, Student_G, save_dir=output_dir)
    _ = Student_G_checkpointer.load(args.student_pretrained)
    ####teacher net##########
    cfg.merge_from_file(args.teacher_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR
    _, Teacher_Head, _ = build_detection_model_GAN_Disting(cfg)
    Teacher_Head.to(cfg.MODEL.DEVICE)
    Teacher_Head_checkpointer = DetectronCheckpointer(cfg, Teacher_Head, save_dir=output_dir)
    _ = Teacher_Head_checkpointer.load(args.teacher_pretrained)


    model = torch.nn.Sequential(*[Student_G,Teacher_Head])





    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference_G_HEAD(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


if __name__ == "__main__":
    main()

    # --student - file
    # configs / GKD / fcos_R_50_FPN_1x.yaml - -teacher - file
    # configs / GKD / fcos_X_101_64x4d_FPN_2x.yaml - -teacher - pretrained / data / training_dir / fcos_R_50_FPN_1x_Head / model_Head_0137500.pth - -student - pretrained / data / training_dir / fcos_R_50_FPN_1x_G / model_G_0137500.pth
    # TEST.IMS_PER_BATCH
    # 4