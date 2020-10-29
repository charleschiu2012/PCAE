#!/bin/bash

dataset_size="840528/210288/210288"

#python ddp_train_img_nice.py \ "Dataparallel" DistributedDataParallel
#OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 ddp_train_img_nice.py \
python ddp_train_img_nice.py \
--gpu_usage 1 \
--dataparallel_mode "Dataparallel" \
--dataset_name "LMNet_ShapeNet_PC" \
--dataset_size "$dataset_size" \
--resample_amount 2048 \
--mode_flag "lm" \
--prior_model "LMNetAE" \
--img_encoder "LMImgEncoder" \
--checkpoint_path "/data/LMNet-data/checkpoint/DDP/ImgNICE" \
--prior_epoch "241" \
--loss_scale_factor 10000 \
--batch_size 32 \
--latent_size 512 \
--epoch_num 300 \
--learning_rate 1e-3 \
--nice_batch_size 32 \
--latent_distribution "normal" \
--mid_dim 128 \
--num_iters 25000 \
--num_sample 64 \
--coupling 4 \
--mask_config 1. \
--nice_epoch 300 \
--project_name "Analogy" \
--run_name "ImgNICE" \
--machine_id "TWCC" \
--step_loss_freq 500 \
--visual_flag