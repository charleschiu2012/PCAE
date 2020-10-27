#!/bin/bash

dataset_size="35022/8762/8762"

#python ddp_train_ae.py \ #--dataparallel_mode "Dataparallel" \
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 ddp_train_ae.py \
--gpu_usage 8 \
--dataparallel_mode "DistributedDataParallel" \
--dataset_name "LMNet_ShapeNet_PC" \
--dataset_size "$dataset_size" \
--resample_amount 2048 \
--mode_flag "ae" \
--prior_model "LMNetAE" \
--checkpoint_path "/data/LMNet-data/checkpoint/DDP/LMNetAE" \
--loss_scale_factor 10000 \
--batch_size 32 \
--latent_size 512 \
--epoch_num 300 \
--learning_rate 5e-4 \
--project_name "Analogy" \
--run_name "Autoencoder" \
--machine_id "TWCC" \
--step_loss_freq 100 \
--visual_flag