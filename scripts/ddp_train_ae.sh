OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 ddp_train_ae.py

