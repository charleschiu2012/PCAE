#!/bin/bash

dataset_size="35022/8762/8762"

python dp_valid_lm_all_view.py \
--gpu_usage 8 \
--dataparallel_mode "Dataparallel" \
--dataset_name "LMNet_ShapeNet_PC" \
--dataset_size "$dataset_size" \
--resample_amount 2048 \
--mode_flag "all_view" \
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
--run_name "ImgEncoder_L1_all_view" \
--machine_id "TWCC" \
--step_loss_freq 500 \
--visual_flag