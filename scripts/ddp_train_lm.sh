#!/bin/bash

dataset_size="840528/210288/210288"

#python ddp_train_lm.py \ #--dataparallel_mode "Dataparallel" \ DistributedDataParallel
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 ddp_train_lm.py \
--gpu_usage 8 \
--dataparallel_mode "DistributedDataParallel" \
--dataset_name "LMNet_ShapeNet_PC" \
--dataset_size "$dataset_size" \
--resample_amount 2048 \
--mode_flag "lm" \
--prior_model "LMNetAE" \
--img_encoder "LMImgEncoder" \
--checkpoint_path "/data/LMNet-data/checkpoint/DDP/LMImgEncoder" \
--prior_epoch "LMNetAE/epoch241.pth" \
--loss_scale_factor 10000 \
--batch_size 32 \
--latent_size 512 \
--epoch_num 300 \
--learning_rate 5e-5 \
--project_name "Analogy" \
--run_name "ImgEncoder" \
--machine_id "TWCC" \
--step_loss_freq 1000 \
--visual_flag