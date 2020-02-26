# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
import os,sys
sys.path.insert(0,'/data/hongwei/face_benchmark')
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
#172.24.42.80
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import torch
import torchvision.transforms as T
from torch.nn.parallel import DataParallel
from maskrcnn_benchmark.config import face_dk_cfg as cfg
from maskrcnn_benchmark.data import make_face_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.inference import inference

from maskrcnn_benchmark.engine import do_face_train_dk_dist_DIV_FC

from maskrcnn_benchmark.modeling.face_reg import build_dist_dk_face_trainer
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, \
    get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.modeling.face_reg import FaceDistributedDataParallel
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

def train(cfg, local_rank, distributed):
    teacher_model,student_model,head = build_dist_dk_face_trainer(cfg,local_rank)
    device = torch.device(cfg.MODEL.DEVICE)
    teacher_model.to(device)
    student_model.to(device)
    if cfg.MODEL.USE_SYNCBN:
        student_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student_model)
    if True:
        # teacher_model = FaceDistributedDataParallel(
        #     teacher_model, device_ids=local_rank, output_device=local_rank,
        #     # this should be removed if we update BatchNorm stats
        #     broadcast_buffers=False,chunk_sizes=None, #[32,56,56,56]
        # )
        teacher_model = DataParallel(teacher_model,device_ids=local_rank,output_device=local_rank[0])

        student_model = FaceDistributedDataParallel(
            student_model, device_ids=local_rank, output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,chunk_sizes=None, #[32,56,56,56]
        )
        head_local_rank=None
        if len(local_rank)==1:
            head_local_rank = local_rank
        head = FaceDistributedDataParallel(
            head, device_ids=head_local_rank, output_device=head_local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )
    teacher_model_head = torch.nn.Sequential(*[teacher_model, head])
    TeacherParam       = torch.load(cfg.MODEL.TEACHERWEIGHT, map_location=torch.device("cpu"))
    TeacherParam['model']['1.module.FC.weights.0'] = TeacherParam['model']['1.module.weights.0']
    TeacherParam['model']['1.module.FC.weights.1'] = TeacherParam['model']['1.module.weights.1']
    TeacherParam['model']['1.module.FC.weights.2'] = TeacherParam['model']['1.module.weights.2']
    TeacherParam['model']['1.module.FC.weights.3'] = TeacherParam['model']['1.module.weights.3']
    load_state_dict(teacher_model_head, TeacherParam.pop("model"))
    teacher_model,head = teacher_model_head[0],teacher_model_head[1]

    # model = torch.nn.Sequential(*[student_model, head])
    model = student_model
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    head_optimizer = make_optimizer(cfg, head)
    head_scheduler = make_lr_scheduler(cfg, head_optimizer)
    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)
    head, head_optimizer = amp.initialize(head, head_optimizer, opt_level=amp_opt_level)


    arguments = {}
    arguments["iteration"] = 0
    output_dir = cfg.OUTPUT_DIR
    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    head_checkpointer = DetectronCheckpointer(
        cfg, head, head_optimizer, head_scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)


    #### init transforms #####
    transforms = T.Compose(
        [
            T.RandomCrop( (cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1]) ),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.RGB_MEAN, std=cfg.INPUT.RGB_STD),
        ]
    )
    data_loader = make_face_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
        transforms=transforms,
    )
    test_period = cfg.SOLVER.TEST_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    divs_nums = cfg.SOLVER.DIVS_NUMS_PER_BATCH
    do_face_train_dk_dist_DIV_FC(
        cfg,
        [model,head,teacher_model],#[model,head],
        data_loader,
        None,
        [optimizer,head_optimizer],
        [scheduler,head_scheduler],
        [checkpointer,head_checkpointer],
        device,
        checkpoint_period,
        test_period,
        arguments,
        divs_nums,
    )
    return model


def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
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
        inference(
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


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--ngpu_shared_fc", type=list, default=1)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    "MASTER_ADDR"
    "MASTER_PORT"
    "RANK"
    "WORLD_SIZE"
    if True:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",#rank=args.local_rank,world_size=size
        )
        synchronize()


    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    proc_gpus = [int(i) for i in args.ngpu_shared_fc]
    model = train(cfg,proc_gpus, args.distributed)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)
if __name__ == "__main__":
    main()

##--nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
## 杀掉所有python进程 ps aux|grep python|grep -v grep|grep -v usr|cut -c 9-15|xargs kill -9
# python tools/Muti_GPUS_Train.py --ngpus_per_node=8 --npgpu_per_proc=1 tools/train_face_netDivFC.py --skip-test --config-file configs/face_reg/face_net_msra_celeb.yaml DATALOADER.NUM_WORKERS 16 OUTPUT_DIR