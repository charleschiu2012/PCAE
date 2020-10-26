#!/bin/bash

train_class_list="airplane, car, chair, lamp, monitor, rifle, sofa"

python dp_valid_ae.py \
--gpu_usage 8 \
--dataparallel_mode "Dataparallel" \
--dataset_name "LMNet_ShapeNet_PC" \
--train_dataset_size 21820 \
--test_dataset_size 5458 \
--valid_dataset_size 5458 \
--resample_amount 2048 \
--train_half_class "$train_class_list" \
--mode_flag "ae" \
--prior_model "LMNetAE" \
--checkpoint_path "/data/LMNet-data/checkpoint/DDP/LMNetAE_half_class" \
--loss_scale_factor 10000 \
--batch_size 32 \
--latent_size 512 \
--epoch_num 300 \
--learning_rate 5e-4 \
--project_name "PCAE" \
--run_name "Autoencoder" \
--machine_id "TWCC" \
--step_loss_freq 50 \
--visual_flag