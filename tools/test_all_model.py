import argparse
import os,sys
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer,make_optimizer_adam
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train_GAN_Disting
from maskrcnn_benchmark.modeling.detector import build_detection_model_GAN_Disting
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--teacher-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--student-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    # parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
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
    parser.add_argument('--clamp', type=float, default=0.03)
    parser.add_argument('--train-D-eopch', type=int, default=2)
    parser.add_argument(
        "--teacher-weights",
        default="",
        metavar="FILE",
        help="path to teacher params",
        type=str,
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()




    # model_G,model_Head,model_Dirc = build_detection_model_GAN_Disting(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    #########===================student net=================#############
    cfg.merge_from_file(args.student_file)
    cfg.merge_from_list(args.opts)

    Student_G, _, _ = build_detection_model_GAN_Disting(cfg)
    Student_G.to(device)

    output_dir_G = cfg.OUTPUT_DIR + '_G'
    save_to_disk = get_rank() == 0
    optimizer_G = make_optimizer(cfg, Student_G)
    scheduler_G = make_lr_scheduler(cfg, optimizer_G)
    checkpointer_G = DetectronCheckpointer(
        cfg, Student_G, optimizer_G, scheduler_G,output_dir_G, save_to_disk
    )
    extra_checkpoint_data_G = checkpointer_G.load(cfg.MODEL.WEIGHT)


    ####teacher net##########
    cfg.merge_from_file(args.teacher_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    Teacher_G, Teacher_Head, model_Dirc = build_detection_model_GAN_Disting(cfg)
    Teacher_G.to(device)
    Teacher_Head.to(device)
    model_Dirc.to(device)

    output_dir_Head = cfg.OUTPUT_DIR+'_Head'
    output_dir_Dirc = cfg.OUTPUT_DIR+'_Dirc'
    save_to_disk = get_rank() == 0
    optimizer_Head = make_optimizer(cfg, Teacher_Head)
    optimizer_Dirc = make_optimizer(cfg, model_Dirc)
    scheduler_Head = make_lr_scheduler(cfg, optimizer_Head)
    scheduler_Dirc = make_lr_scheduler(cfg, optimizer_Dirc)


    checkpointer_Teacher_G = DetectronCheckpointer(
        cfg, Teacher_G,
    )
    checkpointer_Head = DetectronCheckpointer(
        cfg, Teacher_Head, optimizer_Head, scheduler_Head, output_dir_Head, save_to_disk
    )
    checkpointer_Dirc = DetectronCheckpointer(
        cfg, model_Dirc, optimizer_Dirc, scheduler_Dirc, output_dir_Dirc, save_to_disk
    )

    ###########===========test teacher_G  teacher_Head============########



    Teacher_model = torch.nn.Sequential(*[Student_G, Teacher_Head])
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
    from maskrcnn_benchmark.engine.inference import inference_G_HEAD
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference_G_HEAD(
            Teacher_model,
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

    pass


if __name__ == "__main__":
    main()