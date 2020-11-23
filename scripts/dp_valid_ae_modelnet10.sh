#!/bin/bash

dataset_size="3991/908/908"

#python dp_valid_ae.py \ #--dataparallel_mode "Dataparallel" DistributedDataParallel\
#OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 ddp_train_ae.py \
CUDA_VISIBLE_DEVICES=7 \
python dp_valid_ae.py \
--gpu_usage 1 \
--dataparallel_mode "Dataparallel" \
--dataset_name "ModelNet10" \
--dataset_size "$dataset_size" \
--resample_amount 512 \
--mode_flag "ae" \
--prior_model "PointNetAE" \
--checkpoint_path "/data/ModelNet10/checkpoint/PointNetAE" \
--loss_scale_factor 10000 \
--batch_size 512 \
--latent_size 100 \
--epoch_num 300 \
--learning_rate 5e-4 \
--project_name "MNet10" \
--run_name "Autoencoder" \
--machine_id "TWCC" \
--step_loss_freq 1 \
--visual_flag