#!/bin/bash

train_class_list="chair"
#dataset_size="35022/8762/8762"
dataset_size="5422/1356/1356"

#python ddp_train_nice.py \ "Dataparallel" DistributedDataParallel
#OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 ddp_train_nice.py \
CUDA_VISIBLE_DEVICES=0 \
python ddp_train_nice.py \
--gpu_usage 1 \
--dataparallel_mode "Dataparallel" \
--dataset_name "LMNet_ShapeNet_PC" \
--dataset_size "$dataset_size" \
--resample_amount 512 \
--train_half_class "$train_class_list" \
--mode_flag "nice" \
--prior_model "LMNetAE" \
--img_encoder "LMImgEncoder" \
--checkpoint_path "/data/LMNet-data/checkpoint/DP/PCFlow_chair" \
--prior_epoch "DP/LMNetAE_chair/epoch158.pth" \
--loss_scale_factor 10000 \
--batch_size 32 \
--latent_size 10 \
--epoch_num 150 \
--learning_rate 1e-3 \
--nice_lr 5e-7 \
--nice_batch_size 32 \
--latent_distribution "normal" \
--mid_dim 5 \
--num_iters 25000 \
--num_sample 64 \
--coupling 4 \
--mask_config 1. \
--project_name "Analogy_chair" \
--run_name "PCFlow_chair" \
--machine_id "TWCC" \
--step_loss_freq 1 \
--visual_flag