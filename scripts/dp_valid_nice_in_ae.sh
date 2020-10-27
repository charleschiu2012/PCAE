#!/bin/bash

dataset_size="35022/8762/8762"

python dp_valid_nice_in_ae.py \
--gpu_usage 8 \
--dataparallel_mode "Dataparallel" \
--dataset_name "LMNet_ShapeNet_PC" \
--dataset_size "$dataset_size" \
--resample_amount 2048 \
--mode_flag "nice" \
--checkpoint_path "/data/LMNet-data/checkpoint/DDP/NICE_in_AE" \
--loss_scale_factor 10000 \
--batch_size 32 \
--latent_size 512 \
--epoch_num 300 \
--learning_rate 1e-3 \
--nice_batch_size 200 \
--latent_distribution "normal" \
--mid_dim 128 \
--num_iters 25000 \
--num_sample 64 \
--coupling 4 \
--mask_config 1. \
--project_name "Analogy" \
--run_name "NICE_in_AE" \
--machine_id "TWCC" \
--step_loss_freq 50 \
--visual_flag