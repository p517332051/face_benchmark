#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net_GAN_Disting.py \
    --teacher-weights /home/startdt/.torch/models/FCOS_R_50_FPN_1x.pth \
    --skip-test \
    --student-file configs/GKD/fcos_bn_bs16_MNV2_FPN_1x.yaml \
    --teacher-file configs/GKD/fcos_R_50_FPN_1x.yaml \
    DATALOADER.NUM_WORKERS 4 \
    OUTPUT_DIR /data/hongwei/training_dir/FCOS_MV1_RES50_\




#--teacher-weights
#/home/startdt/.torch/models/FCOS_R_50_FPN_1x.pth
#--skip-test
#--student-file
#configs/GKD/fcos_bn_bs16_MNV2_FPN_1x.yaml
#--teacher-file
#configs/GKD/fcos_R_50_FPN_1x.yaml
#DATALOADER.NUM_WORKERS
#2
#OUTPUT_DIR
#/data/hongwei/training_dir/FCOS_MV1_RES50_

#--teacher-weights
#/data/hongwei/training_dir/retinanet_R-101-FPN_1x/model_final.pth
#--skip-test
#--student-file
#configs/retinanet/face_net_msra_celeb.yaml
#--teacher-file
#configs/retinanet/retinanet_R-101-FPN_1x.yaml