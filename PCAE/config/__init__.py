# PCAE/config/__init__
import os
import gc
import torch
from pathlib import Path

from .cuda import CudaConfig
from .dataset import DatasetConfig
from .network import NetworkConfig
from .wandb import WandbConfig
from .nice import NICEConfig

# Comment below code to enable only cpu usage or parallel gpu usage
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
torch.cuda.empty_cache()
gc.collect()

# import random
# import numpy as np
# def set_seeds():
#     np.random.seed(0)
#     random.seed(0)
#     torch.manual_seed(0)
#     torch.cuda.manual_seed(0)
#     torch.cuda.manual_seed_all(0)
#     torch.backends.cudnn.enabled = True
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True

usable_gpu_ids = [0, 1, 2, 3, 4, 5, 6, 7]


class Config:
    def __init__(self, argument):
        # set_seeds()
        self.home_dir = str(Path.home())
        self.tensorboard_dir = self.home_dir + '/tensorboard_runs/'
        self.cuda = CudaConfig(device='cuda' if torch.cuda.is_available() else 'cpu',
                               is_parallel=True,
                               parallel_gpu_ids=usable_gpu_ids[:int(argument.gpu_usage)],
                               dataparallel_mode=argument.dataparallel_mode)  # Dataparallel DistributedDataParallel
        # OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main.py

        dataset_size_list = argument.dataset_size.split('/')
        if argument.train_half_class is not None:
            train_class_list = argument.train_half_class.split(', ')
        else:
            train_class_list = None
        self.dataset = DatasetConfig(dataset_name=argument.dataset_name,
                                     dataset_path=self.home_dir + '/data/LMNet-data/',
                                     # sudo mount tmpfs /eva_data/hdd2/charles/Ramdisk/ -t tmpfs -o size=70G
                                     dataset_size={'train': int(dataset_size_list[0]),
                                                   'test': int(dataset_size_list[1]),
                                                   'valid': int(dataset_size_list[2])},
                                     resample_amount=2048,
                                     train_class=train_class_list,
                                     test_unseen_flag=argument.test_unseen_flag)

        self.network = NetworkConfig(mode_flag=argument.mode_flag,  # ae, lm, vae, nice
                                     img_encoder=argument.img_encoder,  # LMImgEncoder ImgEncoderVAE
                                     prior_model=argument.prior_model,  # PointNetAE LMNetAE
                                     checkpoint_path=self.home_dir + argument.checkpoint_path,
                                     prior_epoch=argument.prior_epoch,
                                     loss_scale_factor=argument.loss_scale_factor,
                                     batch_size=argument.batch_size,
                                     latent_size=argument.latent_size,
                                     z_dim=argument.z_dim,
                                     epoch_num=argument.epoch_num,
                                     learning_rate=argument.learning_rate)
        if argument.mode_flag == 'nice' or (argument.nice_epoch is not None):
            self.nice = NICEConfig(batch_size=argument.nice_batch_size,
                                   latent=argument.latent_distribution,
                                   mid_dim=argument.mid_dim,
                                   num_iters=argument.num_iters,
                                   sample_size=argument.num_sample,
                                   coupling=argument.coupling,
                                   mask_config=argument.mask_config,
                                   nice_epoch=argument.nice_epoch)

        self.wandb = WandbConfig(project_name=argument.project_name,
                                 run_name=argument.run_name,
                                 dir_path=self.home_dir + '/PCAE-TWCC/',
                                 machine_id=argument.machine_id,
                                 step_loss_freq=argument.step_loss_freq,
                                 visual_flag=argument.visual_flag)

    def show_config(self):
        print("Config Setting:")

        data = {'cuda': self.cuda.__dict__,
                'dataset': self.dataset.__dict__,
                'network': self.network.__dict__,
                'wandb': self.wandb.__dict__}

        for k1, v1 in data.items():
            print(k1)
            for k2, v2 in v1.items():
                print('\t{0}: {1}'.format(k2, v2))
            print()
