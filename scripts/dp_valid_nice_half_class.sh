#!/bin/bash

train_class_list="airplane, car, chair, lamp, monitor, rifle, sofa"
dataset_size="21820/5458/5458"

python dp_valid_nice.py \
--gpu_usage 1 \
--dataparallel_mode "Dataparallel" \
--dataset_name "LMNet_ShapeNet_PC" \
--dataset_size "$dataset_size" \
--resample_amount 2048 \
--train_half_class "$train_class_list" \
--mode_flag "nice" \
--prior_model "LMNetAE" \
--img_encoder "LMImgEncoder" \
--checkpoint_path "/data/LMNet-data/checkpoint/DDP/NICE_half_class" \
--prior_epoch "LMNetAE/epoch296.pth" \
--loss_scale_factor 10000 \
--batch_size 512 \
--latent_size 512 \
--epoch_num 300 \
--learning_rate 1e-3 \
--nice_batch_size 512 \
--latent_distribution "normal" \
--mid_dim 128 \
--num_iters 25000 \
--num_sample 64 \
--coupling 4 \
--mask_config 1. \
--project_name "PC_NICE" \
--run_name "NICE_half_class" \
--machine_id "TWCC" \
--step_loss_freq 50 \
--visual_flag