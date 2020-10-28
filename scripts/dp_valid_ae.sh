#!/bin/bash

dataset_size="35022/8762/8762"

python dp_valid_ae.py \
--gpu_usage 1 \
--dataparallel_mode "Dataparallel" \
--dataset_name "LMNet_ShapeNet_PC" \
--dataset_size "$dataset_size" \
--resample_amount 2048 \
--mode_flag "ae" \
--prior_model "LMNetAE" \
--checkpoint_path "/data/LMNet-data/checkpoint/DDP/LMNetAE_half_class" \
--loss_scale_factor 10000 \
--batch_size 32 \
--latent_size 512 \
--epoch_num 300 \
--learning_rate 5e-4 \
--project_name "Analogy" \
--run_name "Autoencoder" \
--machine_id "TWCC" \
--step_loss_freq 200 \
--visual_flag