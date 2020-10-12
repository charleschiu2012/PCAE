import logging
import argparse
import multiprocessing
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader

from ..dataloader import PCDataset
from .networks.loss import PCLoss
from .networks import Network
from .networks.models import PointNetAE, LMNetAE, LMImgEncoder
from ..visualizer import WandbVisualizer, tsne
from ..config import config
from ..utils import model_util


class TrainSession(Network):
    def __init__(self, args, dataloader, model=None, sampler=None):
        if config.network.mode_flag == 'ae':
            super().__init__(model=model, data_loader=dataloader, data_type='train',
                             loss_func=PCLoss().loss, optimizer=config.network.optimizer, epoch=1)
        elif config.network.mode_flag == 'lm':
            super().__init__(model=model, data_loader=dataloader, data_type='train',
                             loss_func='L1Loss', optimizer=config.network.optimizer, epoch=1)

        self._is_scratch = args.scratch
        self._pretrained_epoch = ''
        # self.set_arguments(args)
        # self.prefetcher = DataPrefetcher(self.data_loader)

        self.avg_step_loss = 0.0
        self.avg_epoch_loss = 0.0
        self.latent_list = []
        self.label_list = []
        self.latents = []
        self.latent_ids = []
        self.visualizer = None
        if config.network.mode_flag == 'lm':
            self.prior_model = None
        if config.cuda.dataparallel_mode == 'DistributedDataParallel':
            self.sampler = sampler

    def train(self):
        self.set_model()
        if config.wandb.visual_flag:
            self.visualizer = WandbVisualizer(job_type='train', model=self.model)

        for epoch_idx in range(self._epoch - 1, config.network.epoch_num):
            logging.info('Start training epoch %d' % (epoch_idx + 1))
            if config.network.mode_flag == 'ae':
                self.model.train()
            elif config.network.mode_flag == 'lm':
                self.prior_model.eval()
                self.model.train()

            if config.cuda.dataparallel_mode == 'DistributedDataParallel':
                self.sampler.set_epoch(self._epoch)

            if config.network.mode_flag == 'ae':
                for idx, (inputs_pc, targets, pc_ids) in tqdm(enumerate(self.get_data())):
                    # try:
                    self.optimizer.zero_grad()
                    latents, predicts = self.model(inputs_pc)

                    loss = self.loss_func(predicts, targets) * config.network.loss_scale_factor

                    loss.backward()
                    self.optimizer.step()

                    self.log_step_loss(loss=loss.item(), step_idx=idx + 1)
                    self.avg_step_loss = 0
                    # except:
                    #     for i in range(inputs_pc.shape[0]):
                    #         print(inputs_pc.shape)
                    #         print(pc_id[i])
                    #     exit(1)

                    ## draw tsne
                    # self.latent_list = tsne.add_tsne_data(latent_list=self.latent_list, latent_data=latents)
                    # self.label_list = tsne.add_tsne_label(label_list=self.label_list, label=pc_id)
                    # self.latents.extend(latents.detach().cpu().numpy())
                    # del latents
                    # self.latent_ids.extend(pc_ids)

                self.save_model()
                self.log_epoch_loss()
                # tsne.visualize_tsne(latent_list=self.latent_list, label_list=self.label_list, job_type='train')
                # self.save_latent()
                self._epoch += 1

            elif config.network.mode_flag == 'lm':
                for idx, (inputs_img, inputs_pc, targets, _, _) in tqdm(enumerate(self.get_data())):
                    self.optimizer.zero_grad()
                    with torch.no_grad():
                        latent_pc, _ = self.prior_model(inputs_pc)
                    latent_img = self.model(inputs_img)
                    loss = torch.nn.L1Loss()(latent_img, latent_pc) * config.network.loss_scale_factor

                    loss.backward()
                    self.optimizer.step()

                    self.log_step_loss(loss=loss.item(), step_idx=idx + 1)
                    self.avg_step_loss = 0

                self.save_model()
                self.log_epoch_loss()
                self._epoch += 1

        if config.wandb.visual_flag and config.network.mode_flag == 'ae':
            self.visualizer.log_point_clouds(self.tensor_to_numpy(predicts[-1].view(-1, 3)),
                                             self.tensor_to_numpy(targets[-1].view(-1, 3)))

        if config.cuda.dataparallel_mode == 'DistributedDataParallel':
            model_util.cleanup()

    def set_arguments(self, args):
        self._pretrained_epoch = args.pretrained_epoch

    def set_model(self):
        models = {'PointNetAE': PointNetAE, 'LMNetAE': LMNetAE, 'LMImgEncoder': LMImgEncoder}
        if config.network.mode_flag == 'ae':
            self.model = models[config.network.prior_model](config.dataset.resample_amount)
        elif config.network.mode_flag == 'lm':
            self.model = models[config.network.img_encoder](config.network.latent_size)
            self.prior_model = models[config.network.prior_model](config.dataset.resample_amount)
            prior_model_path = '%s/%s/epoch%.3d.pth' % (config.network.checkpoint_path,
                                                        config.network.prior_model,
                                                        int(config.network.prior_epoch))
            self.prior_model = model_util.set_model_device(self.prior_model)
            self.prior_model = model_util.set_model_parallel_gpu(self.prior_model)
            if config.cuda.dataparallel_mode == 'Dataparallel':
                self.prior_model.load_state_dict(state_dict=torch.load(f=prior_model_path))
            elif config.cuda.dataparallel_mode == 'DistributedDataParallel':
                # Use a barrier() to make sure that process 1 loads the model after process
                # 0 saves it.
                torch.distributed.barrier()
                # configure map_location properly
                map_location = {'cuda:%d' % 0: 'cuda:%d' % config.cuda.rank}
                self.prior_model.load_state_dict(state_dict=torch.load(f=prior_model_path, map_location=map_location))

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
        # self.avg_epoch_loss /= self.steps_in_an_epoch()
        if config.cuda.dataparallel_mode == 'Dataparallel':
            self.avg_epoch_loss /= PCDataset(self._data_type).__len__()
        elif config.cuda.dataparallel_mode == 'DistributedDataParallel':
            self.avg_epoch_loss /= (PCDataset(self._data_type).__len__() / len(config.cuda.parallel_gpu_ids))

        logging.info('Logging Epoch Loss...')
        if config.wandb.visual_flag:
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, train_epoch_loss=self.avg_epoch_loss)
            self.avg_epoch_loss = .0

    def save_latent(self):
        print(len(self.latents))
        for idx, (latent, pc_id) in enumerate(zip(self.latents, self.latent_ids)):
            import os
            save_path = None
            if config.network.mode_flag == 'ae':
                save_path = os.path.join(config.network.checkpoint_path + '/ae_latent', pc_id)
            elif config.network.mode_flag == 'lm':
                save_path = os.path.join(config.network.checkpoint_path + '/lm_latent', pc_id)
            os.makedirs(save_path, exist_ok=True)
            np.save('{}/latent'.format(save_path), latent)


def train():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    config.show_config()

    parser = argparse.ArgumentParser()
    parser.add_argument('-pte', '--pretrained_epoch', type=str,
                        help='Use a pretrained model to continue training.')
    parser.add_argument('-s', '--scratch', action='store_true',
                        help='Train model from scratch.')

    train_dataset = PCDataset('train')

    if config.cuda.dataparallel_mode == 'Dataparallel':
        argument = parser.parse_args()
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=config.network.batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      num_workers=multiprocessing.cpu_count() * 5)
        train_session = TrainSession(args=argument, dataloader=train_dataloader)
        train_session.train()

    elif config.cuda.dataparallel_mode == 'DistributedDataParallel':
        parser.add_argument("--local_rank", type=int)
        argument = parser.parse_args()
        torch.cuda.set_device(argument.local_rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        num_replicas=config.cuda.world_size[0],
                                                                        rank=config.cuda.rank[0])
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=config.network.batch_size,
                                      shuffle=(train_sampler is None),
                                      pin_memory=True,
                                      sampler=train_sampler,
                                      num_workers=8,
                                      worker_init_fn=np.random.seed(0))
        train_session = TrainSession(args=argument, dataloader=train_dataloader, sampler=train_sampler)
        train_session.train()
