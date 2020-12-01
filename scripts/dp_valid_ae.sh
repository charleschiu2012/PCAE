#!/bin/bash

train_class_list="chair"
#dataset_size="35022/8762/8762"
dataset_size="5422/1356/1356"

CUDA_VISIBLE_DEVICES=0 \
python dp_valid_ae.py \
--gpu_usage 1 \
--dataparallel_mode "Dataparallel" \
--dataset_name "LMNet_ShapeNet_PC" \
--dataset_size "$dataset_size" \
--resample_amount 512 \
--train_half_class "$train_class_list" \
--mode_flag "ae" \
--prior_model "LMNetAE" \
--checkpoint_path "/data/LMNet-data/checkpoint/DP/LMNetAE_chair" \
--loss_scale_factor 10000 \
--batch_size 512 \
--latent_size 10 \
--epoch_num 150 \
--learning_rate 5e-4 \
--project_name "Analogy_chair" \
--run_name "Autoencoder_chair" \
--machine_id "TWCC" \
--step_loss_freq 1 \
--visual_flag