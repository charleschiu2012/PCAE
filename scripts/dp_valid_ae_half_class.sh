#!/bin/bash

train_class_list="airplane, car, chair, lamp, monitor, rifle, sofa"
dataset_size="21820/5458/5458"

python dp_valid_ae.py \
--gpu_usage 1 \
--dataparallel_mode "Dataparallel" \
--dataset_name "LMNet_ShapeNet_PC" \
--dataset_size "$dataset_size" \
--resample_amount 2048 \
--train_half_class "$train_class_list" \
--mode_flag "ae" \
--prior_model "LMNetAE" \
--checkpoint_path "/data/LMNet-data/checkpoint/DDP/LMNetAE_half_class" \
--loss_scale_factor 10000 \
--batch_size 512 \
--latent_size 512 \
--epoch_num 300 \
--learning_rate 5e-4 \
--project_name "PCAE" \
--run_name "Autoencoder" \
--machine_id "TWCC" \
--step_loss_freq 50 \
--visual_flag