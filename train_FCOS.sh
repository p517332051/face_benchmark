#!/usr/bin/env bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net.py \
    --skip-test \
    --config-file configs/retinanet/retinanet_R-101-FPN_1x.yaml \
    DATALOADER.NUM_WORKERS 2 \
    OUTPUT_DIR /data/hongwei/training_dir/retinanet_R-101-FPN_1x