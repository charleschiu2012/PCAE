# PCAE/config/__init__
import os
import gc
import torch
from pathlib import Path

from .cuda import CudaConfig
from .dataset import DatasetConfig
from .network import NetworkConfig
from .flow import FlowConfig
from .wandb import WandbConfig

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


class Config:
    def __init__(self):
        # set_seeds()
        self.home_dir = str(Path.home())
        # self.data_dir = '/dev/shm/'
        self.tensorboard_dir = self.home_dir + '/tensorboard_runs/'
        self.cuda = CudaConfig(device='cuda' if torch.cuda.is_available() else 'cpu',
                               is_parallel=True,
                               parallel_gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
                               # parallel_gpu_ids=[0, 1],
                               dataparallel_mode='Dataparallel')  # Dataparallel DistributedDataParallel
        # OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main.py

        self.dataset = DatasetConfig(dataset_name='LMNet_ShapeNet_PC',
                                     dataset_path=self.home_dir + '/data/LMNet-data/',
                                     # sudo mount tmpfs /eva_data/hdd2/charles/Ramdisk/ -t tmpfs -o size=70G
                                     dataset_size={'train': 35022 * 24, 'test': 8762 * 24, 'valid': 8762 * 24},
                                     # dataset_size={'train': 50, 'test': 50, 'valid': 50},
                                     # LM Dataset
                                     # dataset_size={'train': 35022, 'test': 8762, 'valid': 8762},  # AE Dataset
                                     resample_amount=2048,
                                     # not_train_class=['airplane', 'bench', 'cabinet']
                                     )
        # LMNetAE_dataset_size' = {'train': 35022, 'test': 8762, 'valid': 8762}

        self.network = NetworkConfig(mode_flag='lm',  # ae, lm, vae
                                     img_encoder="ImgEncoderVAE",  # LMImgEncoder ImgEncoderVAE
                                     prior_model="LMNetAE",  # PointNetAE LMNetAE
                                     # checkpoint_path=self.home_dir + 'data/LMNet-data/checkpoint/',
                                     checkpoint_path=self.home_dir + '/data/LMNet-data/checkpoint/DDP',
                                     prior_epoch='300',
                                     loss_function='cd',  # emd cd emd+cd l1
                                     loss_scale_factor=10000,
                                     batch_size=32,  # ae:32 lm:24
                                     latent_size=512,
                                     z_dim=512,
                                     epoch_num=300,
                                     optimizer='Adam',
                                     # learning_rate=5e-4,  # ae
                                     learning_rate=5e-5,  # lm
                                     momentum=0.9)

        self.flow = FlowConfig(batch_size=15,
                               ae_dataset_size={'train': 35022, 'test': 8762, 'valid': 8762},
                               lm_dataset_size={'train': 35022 * 24, 'test': 8762 * 24, 'valid': 8762 * 24},
                               train_class=['sofa'],
                               iter=2 * 10e4,
                               n_flow=32,
                               n_block=4,
                               lu_flag=True,
                               affine_flag=True,
                               n_bits=5,
                               lr=1e-4,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
                               temp=0.7,
                               n_sample=20)

        self.wandb = WandbConfig(project_name='PCVAE',
                                 run_name='{}'.format(self.network.img_encoder),
                                 dir_path=self.home_dir + '/PCAE-TWCC/',
                                 machine_id="TWCC",
                                 step_loss_freq=500,
                                 visual_flag=True)

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


config = Config()
