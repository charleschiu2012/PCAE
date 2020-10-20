import logging
import argparse
import multiprocessing
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.distributions as distributions

from ..dataloader import FlowDataset
from .networks import Network
from .networks.models import NICE
from ..visualizer import WandbVisualizer, tsne
from ..config import config
from ..utils import model_util


class TrainNICESession(Network):
    def __init__(self, args, dataloader, model=None, sampler=None):
        super().__init__(model=model, data_loader=dataloader, data_type='train',
                         loss_func="None", optimizer=config.network.optimizer, epoch=1)

        # self._is_scratch = args.scratch
        self._is_scratch = True
        self._pretrained_epoch = ''

        self.avg_step_loss = 0.0
        self.avg_epoch_loss = 0.0
        self.latent_list = []
        self.label_list = []
        self.latents = []
        self.latent_ids = []
        self.visualizer = None
        self.prior_model = None
        if config.cuda.dataparallel_mode == 'DistributedDataParallel':
            self.sampler = sampler

        self.full_dim = 512
        self.hidden = 5
        if config.nice.latent == 'normal':
            self.prior = torch.distributions.Normal(torch.tensor(0.).cuda(), torch.tensor(1.).cuda())

    def train(self):
        self.set_model()
        if config.wandb.visual_flag:
            self.visualizer = WandbVisualizer(job_type='train', model=self.model)

        for epoch_idx in range(self._epoch - 1, config.network.epoch_num):
            logging.info('Start training epoch %d' % (epoch_idx + 1))
            self.model.train()

            if config.cuda.dataparallel_mode == 'DistributedDataParallel':
                self.sampler.set_epoch(self._epoch)

            for idx, (ae_latents, latent_ids) in tqdm(enumerate(self.get_data())):
                self.optimizer.zero_grad()
                loss = -self.model(ae_latents).mean()

                loss.backward()
                self.optimizer.step()

                self.log_step_loss(loss=loss.item(), step_idx=idx + 1)
                self.avg_step_loss = 0

            self.save_model()
            self.log_epoch_loss()
            self._epoch += 1

        if config.cuda.dataparallel_mode == 'DistributedDataParallel':
            model_util.cleanup()

    def set_arguments(self, args):
        self._pretrained_epoch = args.pretrained_epoch

    def set_model(self):
        self.model = NICE(prior=self.prior,
                          coupling=config.nice.coupling,
                          in_out_dim=self.full_dim,
                          mid_dim=config.nice.mid_dim,
                          hidden=self.hidden,
                          mask_config=config.nice.mask_config)
        self.model = model_util.set_model_device(self.model)
        self.model = model_util.set_model_parallel_gpu(self.model)
        self._epoch = model_util.load_model_pretrain(self.model, self._pretrained_epoch, self._is_scratch)

    def log_step_loss(self, loss, step_idx):
        self.avg_step_loss += loss
        self.avg_epoch_loss += loss

        if step_idx % config.wandb.step_loss_freq == 0:
            self.avg_step_loss /= config.wandb.step_loss_freq
            logging.info('Epoch %d, %d Step, loss = %.6f' % (self._epoch, step_idx, self.avg_step_loss))

            if config.wandb.visual_flag:
                self.visualizer.log_step_loss(step_idx=step_idx, step_loss=self.avg_step_loss)

    def log_epoch_loss(self):
        if config.cuda.dataparallel_mode == 'Dataparallel':
            self.avg_epoch_loss /= FlowDataset(self._data_type).__len__()
        elif config.cuda.dataparallel_mode == 'DistributedDataParallel':
            self.avg_epoch_loss /= (FlowDataset(self._data_type).__len__() / len(config.cuda.parallel_gpu_ids))

        logging.info('Logging Epoch Loss...')
        if config.wandb.visual_flag:
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, train_epoch_loss=self.avg_epoch_loss)
            self.avg_epoch_loss = .0


def train_nice():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    config.show_config()

    parser = argparse.ArgumentParser()
    parser.add_argument('-pte', '--pretrained_epoch', type=str,
                        help='Use a pretrained model to continue training.')
    parser.add_argument('-s', '--scratch', action='store_true',
                        help='Train model from scratch.')

    train_dataset = FlowDataset('train')

    if config.cuda.dataparallel_mode == 'Dataparallel':
        argument = parser.parse_args()
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=config.nice.batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      num_workers=multiprocessing.cpu_count() * 5)
        train_session = TrainNICESession(args=argument, dataloader=train_dataloader)
        train_session.train()

    elif config.cuda.dataparallel_mode == 'DistributedDataParallel':
        parser.add_argument("--local_rank", type=int)
        argument = parser.parse_args()
        torch.cuda.set_device(argument.local_rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        num_replicas=config.cuda.world_size[0],
                                                                        rank=config.cuda.rank[0])
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=config.nice.batch_size,
                                      shuffle=(train_sampler is None),
                                      pin_memory=True,
                                      sampler=train_sampler,
                                      num_workers=8,
                                      worker_init_fn=np.random.seed(0))
        train_session = TrainNICESession(args=argument, dataloader=train_dataloader, sampler=train_sampler)
        train_session.train()
