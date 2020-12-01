#!/bin/bash

#dataset_size="840528/210288/210288"
train_class_list="chair"
dataset_size="130128/32544/32544"

CUDA_VISIBLE_DEVICES=1,2 \
python dp_valid_lm.py \
--gpu_usage 2 \
--dataparallel_mode "Dataparallel" \
--dataset_name "LMNet_ShapeNet_PC" \
--dataset_size "$dataset_size" \
--train_half_class "$train_class_list" \
--resample_amount 512 \
--mode_flag "lm" \
--prior_model "LMNetAE" \
--img_encoder "LMImgEncoder" \
--checkpoint_path "/data/LMNet-data/checkpoint/DP/ImgEncoder_L1_chair" \
--prior_epoch "DP/LMNetAE_chair/epoch158.pth" \
--loss_scale_factor 10000 \
--batch_size 512 \
--latent_size 10 \
--epoch_num 150 \
--learning_rate 5e-4 \
--project_name "Analogy_chair" \
--run_name "ImgEncoder_L1_chair" \
--machine_id "TWCC" \
--step_loss_freq 1 \
--visual_flag