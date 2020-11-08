#!/bin/bash

dataset_size="840528/210288/210288"

python dp_valid_lm.py \
--gpu_usage 8 \
--dataparallel_mode "Dataparallel" \
--dataset_name "LMNet_ShapeNet_PC" \
--dataset_size "$dataset_size" \
--resample_amount 2048 \
--mode_flag "lm" \
--prior_model "LMNetAE" \
--img_encoder "LMImgEncoder" \
--checkpoint_path "/data/LMNet-data/checkpoint/DDP/ImgEncoder_L1" \
--prior_epoch "LMNetAE/epoch241.pth" \
--loss_scale_factor 10000 \
--batch_size 512 \
--latent_size 512 \
--epoch_num 300 \
--learning_rate 5e-4 \
--project_name "Analogy" \
--run_name "ImgEncoder_L1" \
--machine_id "TWCC" \
--step_loss_freq 200 \
--visual_flag