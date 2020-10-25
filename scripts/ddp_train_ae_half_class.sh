#!/bin/bash

train_class_list="airplane, car, chair, lamp, monitor, rifle, sofa"

#python ddp_train_ae.py \ Dataparallel DistributedDataParallel
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 ddp_train_ae.py \
--gpu_usage 8 \
--dataparallel_mode "DistributedDataParallel" \
--dataset_name "LMNet_ShapeNet_PC" \
--train_dataset_size 35022 \
--test_dataset_size 8762 \
--valid_dataset_size 8762 \
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