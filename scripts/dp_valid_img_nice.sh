#!/bin/bash

#dataset_size="840528/210288/210288"
train_class_list="chair"
dataset_size="130128/32544/32544"

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python dp_valid_img_nice.py \
--gpu_usage 4 \
--dataparallel_mode "Dataparallel" \
--dataset_name "LMNet_ShapeNet_PC" \
--dataset_size "$dataset_size" \
--resample_amount 512 \
--train_half_class "$train_class_list" \
--mode_flag "img_nice" \
--prior_model "LMNetAE" \
--img_encoder "LMImgEncoder" \
--checkpoint_path "/data/LMNet-data/checkpoint/DP/ImgNICE_encoder_chair" \
--prior_epoch "DP/LMNetAE_chair/epoch158.pth" \
--loss_scale_factor 10000 \
--batch_size 512 \
--latent_size 10 \
--epoch_num 150 \
--learning_rate 5e-4 \
--nice_batch_size 512 \
--nice_lr 1e-3 \
--latent_distribution "normal" \
--mid_dim 5 \
--num_iters 25000 \
--num_sample 64 \
--coupling 4 \
--mask_config 1. \
--nice_epoch "PCFlow/epoch300.pth" \ #TODO
--project_name "Analogy_chair" \
--run_name "ImgNICE_encoder_chair" \
--machine_id "TWCC" \
--step_loss_freq 1 \
--visual_flag