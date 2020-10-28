import torch
import numpy as np
from pathlib import Path
from PCAE.config.cuda import CudaConfig
from PCAE.config.dataset import DatasetConfig
from PCAE.config.network import NetworkConfig
from PCAE.config.wandb import WandbConfig
from PCAE.config.nice import NICEConfig

import torch
import time
from torch.utils.data import DataLoader

from PCAE.dataloader import PCDataset


class Config:
    def __init__(self):
        # set_seeds()
        self.home_dir = str(Path.home())
        # self.data_dir = '/dev/shm/'
        self.tensorboard_dir = self.home_dir + '/tensorboard_runs/'
        self.cuda = CudaConfig(device='cuda' if torch.cuda.is_available() else 'cpu',
                               is_parallel=True,
                               parallel_gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
                               dataparallel_mode='DistributedDataParallel')

        self.dataset = DatasetConfig(dataset_name='LMNet_ShapeNet_PC',
                                     dataset_path=self.home_dir + '/data/LMNet-data/',
                                     # sudo mount tmpfs /eva_data/hdd2/charles/Ramdisk/ -t tmpfs -o size=70G
                                     #                                      dataset_size={'train': 35022 * 24, 'test': 8762 * 24, 'valid': 8762 * 24},
                                     # LM Dataset
                                     dataset_size={'train': 35022, 'test': 8762, 'valid': 8762},  # AE Dataset
                                     resample_amount=2048,
                                     train_class=None,
                                     test_unseen_flag=False)

        self.network = NetworkConfig(mode_flag='lm',  # ae, lm, vae, nice
                                     img_encoder="ImgEncoderVAE",  # LMImgEncoder ImgEncoderVAE
                                     prior_model="LMNetAE",  # PointNetAE LMNetAE
                                     checkpoint_path=self.home_dir + '/data/LMNet-data/checkpoint/DDP/LMNetAE_half_class',
                                     prior_epoch='296',
                                     loss_scale_factor=10000,
                                     batch_size=200,  # ae:32 lm:24
                                     latent_size=512,
                                     z_dim=512,
                                     epoch_num=300,
                                     learning_rate=5e-4)

        self.wandb = WandbConfig(project_name='PCAE',
                                 run_name='{}'.format('Autoencoder'),
                                 dir_path=self.home_dir + '/PCAE-TWCC/',
                                 machine_id="TWCC",
                                 step_loss_freq=500,
                                 visual_flag=False)

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

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    for num_workers in range(5, 35):
        torch.cuda.set_device(config.cuda.rank[0])
        train_dataset = PCDataset(config=config, split_dataset_type='train')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        num_replicas=config.cuda.world_size[0],
                                                                        rank=config.cuda.rank[0])
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=config.network.batch_size,
                                      shuffle=(train_sampler is None),
                                      pin_memory=True,
                                      sampler=train_sampler,
                                      num_workers=num_workers,
                                      worker_init_fn=np.random.seed(0))

        start = time.time()
        for batch_idx, (_) in enumerate(train_dataloader):  # 不断load
            pass
        end = time.time()
        if config.cuda.rank[0] == 0:
            print("Finish with:{} second, num_workers={}, pin_memory={}, mode_flag={}".format(end - start,
                                                                                              num_workers, 'True',
                                                                                              config.network.mode_flag))

    for num_workers in range(5, 35):
        torch.cuda.set_device(config.cuda.rank[0])
        train_dataset = PCDataset(config=config, split_dataset_type='train')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        num_replicas=config.cuda.world_size[0],
                                                                        rank=config.cuda.rank[0])
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=config.network.batch_size,
                                      shuffle=(train_sampler is None),
                                      pin_memory=False,
                                      sampler=train_sampler,
                                      num_workers=num_workers,
                                      worker_init_fn=np.random.seed(0))

        start = time.time()
        for batch_idx, (_) in enumerate(train_dataloader):  # 不断load
            pass
        end = time.time()
        if config.cuda.rank[0] == 0:
            print("Finish with:{} second, num_workers={}, pin_memory={}, mode_flag={}".format(end - start,
                                                                                              num_workers, 'False',
                                                                                              config.network.mode_flag))
