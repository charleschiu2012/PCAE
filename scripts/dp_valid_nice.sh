#!/bin/bash

dataset_size="35022/8762/8762"

python dp_valid_nice.py \
--gpu_usage 1 \
--dataparallel_mode "Dataparallel" \
--dataset_name "LMNet_ShapeNet_PC" \
--dataset_size "$dataset_size" \
--resample_amount 2048 \
--mode_flag "nice" \
--prior_model "LMNetAE" \
--img_encoder "LMImgEncoder" \
--checkpoint_path "/data/LMNet-data/checkpoint/DDP/NICE" \
--prior_epoch "" \
--loss_scale_factor 10000 \
--batch_size 32 \
--latent_size 512 \
--epoch_num 300 \
--learning_rate 1e-3 \
--nice_batch_size 32 \
--latent_distribution "normal" \
--mid_dim 128 \
--num_iters 25000 \
--num_sample 64 \
--coupling 4 \
--mask_config 1. \
--project_name "Analogy" \
--run_name "NICE" \
--machine_id "TWCC" \
--step_loss_freq 200 \
--visual_flag