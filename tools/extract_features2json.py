# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import face_cfg as cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.face_reg import build_face_trainer
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.face_evaluation import rank_1,get_transform,creat_face_json
# Check if we can enable mixed-precision via apex.amp
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')
def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_face_trainer(cfg)
    model.to(cfg.MODEL.DEVICE)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    #############============ face val ================##########
    # ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    BACKBONE_PATH = os.path.join(cfg.OUTPUT_DIR,)
    BACKBONE_ACC = {}
    for BACKBONE in os.listdir(BACKBONE_PATH):
        ckpt = os.path.join(BACKBONE_PATH,BACKBONE)
        _ = checkpointer.load(ckpt, use_latest=False)
        TEST_PATH_LIST = cfg.DATASETS.TEST_PATH_LIST
        TESTS_ACC = {}

        Q_IMG_PATH = "/data/hongwei/face_benchmark/10_11楼层人脸_align"
        G_IMG_PATH = "/data/hongwei/face_benchmark/魔镜人脸抓拍_二号设备_all_align"
        pic_save_path = "/data/hongwei/face_benchmark/tools/IMGLIST_ALL"
        creat_face_json(model,Q_IMG_PATH,G_IMG_PATH,True, cfg.MODEL.DEVICE, cfg.INPUT.SIZE_TRAIN, cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS, get_transform(cfg),
                                     pic_save_path,cfg.SOLVER.IMS_PER_BATCH, 'test')










if __name__ == "__main__":
    main()
