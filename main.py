import open3d
import os
from PCAE.jobs import train, test, validate, train_vae, train_nice


if __name__ == '__main__':
    # train()
    # validate()
    # test(82)
    # train_vae()
    train_nice()
    # test(140)  # single gpu ep150
    # test(96)  # multi gpu ep100 LMNetAE
    # test(100)  # single gpu ep100 PointNetAE
    # test(7)  # LMImgEncoder ep60 512
    # test(3)  # LMImgEncoder distributed

    # if config.network.mode_flag == 'ae':
    # elif config.network.mode_flag == 'lm':
    #
    # if config.cuda.dataparallel_mode == 'Dataparallel':
    # elif config.cuda.dataparallel_mode == 'DistributedDataParallel':
    # OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8  main.py
