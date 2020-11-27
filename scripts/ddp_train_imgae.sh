#!/bin/bash

dataset_size="840528/210288/210288"

#python ddp_train_imgae.py \ Dataparallel  DistributedDataParallel
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 ddp_train_imgae.py \
--gpu_usage 8 \
--dataparallel_mode "DistributedDataParallel" \
--dataset_name "LMNet_ShapeNet_PC" \
--dataset_size "$dataset_size" \
--resample_amount 2048 \
--mode_flag "lm" \
--prior_model "LMNetAE" \
--img_encoder "LMImgEncoder" \
--checkpoint_path "/data/LMNet-data/checkpoint/DDP/ImgAE_MSE_SSIM" \
--batch_size 32 \
--latent_size 512 \
--epoch_num 100 \
--learning_rate 5e-5 \
--project_name "Analogy" \
--run_name "ImgAE_MSE_SSIM" \
--machine_id "TWCC" \
--step_loss_freq 1 \
--visual_flag