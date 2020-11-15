import argparse
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader

from PCAE.config import Config
from PCAE.dataloader import PCDataset
from PCAE.loss import KLDLoss, chamfer_distance_loss
from PCAE.networks import Network
from PCAE.models import LMNetAE, ImgEncoderVAE
from PCAE.visualizer import WandbVisualizer
from PCAE.utils import ModelUtil

parser = argparse.ArgumentParser()

'''cuda
'''
parser.add_argument('--gpu_usage', type=int, required=True, default=8,
                    help='How many gpu you want use')
parser.add_argument('--dataparallel_mode', type=str, required=True,
                    help='Which mode of dataparallel')
parser.add_argument("--local_rank", type=int)
'''dataset
'''
parser.add_argument('--dataset_name', type=str, required=True, default='LMNet_ShapeNet_PC',
                    help='The name of the dataset')
parser.add_argument('--dataset_size', type=str, required=True,
                    help='The sizes of split dataset')
parser.add_argument('--resample_amount', type=int, required=True, default=2048,
                    help='The num of points to sample from original point cloud')
parser.add_argument('--train_half_class', type=str,
                    help='Train with half of the classes')
parser.add_argument('--test_unseen_flag', action='store_true',
                    help='Use this flag to test unseen classes')
'''network
'''
parser.add_argument('--mode_flag', type=str, required=True,
                    help='Mode to train')
parser.add_argument('--img_encoder', type=str, required=True,
                    help='Which Image encoder')
parser.add_argument('--prior_model', type=str, required=True,
                    help='Which point cloud autoencoder')
parser.add_argument('--checkpoint_path', type=str, required=True,
                    help='Where to store/load weights')
parser.add_argument('--prior_epoch', type=str, required=True,
                    help='Which epoch of autoencoder to use to ImgEncoder')
parser.add_argument('--loss_scale_factor', type=int, required=True, default=10000,
                    help='Scale your loss')
parser.add_argument('--batch_size', type=int, required=True, default=32,
                    help='Batch size of point cloud or image')
parser.add_argument('--latent_size', type=int, required=True, default=512,
                    help='Size of latent')
parser.add_argument('--z_dim', type=int, required=True, default=512,
                    help='Size of vae latent')
parser.add_argument('--epoch_num', type=int, required=True, default=300,
                    help='How many epoch to train')
parser.add_argument('--learning_rate', type=float, required=True, default=5e-5,
                    help='Learning rate')
'''wandb
'''
parser.add_argument('--project_name', type=str, required=True,
                    help='Project name to log in wandb')
parser.add_argument('--run_name', type=str, required=True,
                    help='Run name to log under project')
parser.add_argument('--machine_id',
                    help='Which machine')
parser.add_argument('--step_loss_freq', type=int, required=True,
                    help='How many steps for log step loss one time')
parser.add_argument('--visual_flag', action='store_true',
                    help='Use wandb or not')

argument = parser.parse_args()
config = Config(argument)


class VAETrainSession(Network):
    def __init__(self, dataloader, model=None, sampler=None):
        super().__init__(config=config, data_loader=dataloader, data_type='train', epoch=1, model=model)
        self._pretrained_epoch = ''
        self._is_scratch = False

        self.avg_step_loss = 0.0
        self.avg_epoch_loss = 0.0
        self.avg_step_cd_loss = 0.0
        self.avg_epoch_cd_loss = 0.0
        self.avg_step_kld_loss = 0.0
        self.avg_epoch_kld_loss = 0.0
        self.data_length = 0
        self.prior_model = None
        self.pc_decoder = None
        self.model_util = ModelUtil(config=config)
        self.visualizer = None
        if config.cuda.dataparallel_mode == 'DistributedDataParallel':
            self.sampler = sampler

    def train(self):
        self.set_model()
        if ((argument.local_rank is not None) and config.cuda.rank[0] == 0) and config.wandb.visual_flag:
            self.visualizer = WandbVisualizer(config=config, job_type='train', model=self.model)
        elif (argument.local_rank is None) and config.wandb.visual_flag:
            self.visualizer = WandbVisualizer(config=config, job_type='train', model=self.model)

        self.model.train()
        self.prior_model.eval()
        self.pc_decoder.eval()
        for epoch_idx in range(self._epoch - 1, config.network.epoch_num):
            logging.info('Start training epoch %d' % (epoch_idx + 1))

            if config.cuda.dataparallel_mode == 'DistributedDataParallel':
                self.sampler.set_epoch(self._epoch)

            final_step = 0
            for idx, (inputs_img, inputs_pc, targets, _, _) in enumerate(self.get_data()):
                final_step = idx
                self.data_length += len(inputs_pc)

                self.optimizer.zero_grad()
                with torch.no_grad():
                    latent_pc, _ = self.prior_model(inputs_pc)
                latent_img, mu, log_var = self.model(inputs_img)
                with torch.no_grad():
                    re_imgs = self.pc_decoder(latent_img)

                kld_loss = KLDLoss(mu, log_var)
                cd_loss = chamfer_distance_loss(re_imgs, targets)
                loss = kld_loss + cd_loss

                loss.backward()
                self.optimizer.step()

                self.log_step_loss(sum_loss=loss.item() * len(inputs_pc),
                                   cd_loss=cd_loss.item() * len(inputs_pc),
                                   kld_loss=kld_loss.item() * len(inputs_pc),
                                   step_idx=idx + 1)
                self.avg_step_loss = .0
                self.avg_step_cd_loss = .0
                self.avg_step_kld_loss = .0

            logging.info('Epoch %d, %d Step' % (self._epoch, final_step))
            # self.save_model()
            self.log_epoch_loss()
            self.avg_epoch_loss = .0
            self.avg_epoch_cd_loss = .0
            self.avg_epoch_kld_loss = .0
            self.data_length = 0
            self._epoch += 1

        if config.cuda.dataparallel_mode == 'DistributedDataParallel':
            self.model_util.cleanup()

    def set_model(self):
        """Img Encoder
        """
        self.model = ImgEncoderVAE(latent_size=config.network.latent_size, z_dim=config.network.z_dim)
        self.model = self.model_util.set_model_device(self.model)
        self.model = self.model_util.set_model_parallel_gpu(self.model)
        self._epoch = self.model_util.load_model_pretrain(self.model, self._pretrained_epoch, self._is_scratch)
        '''Prior Model
        '''
        self.prior_model = LMNetAE(config.dataset.resample_amount)
        self.prior_model = self.model_util.load_trained_model(self.prior_model, config.network.prior_epoch)
        '''PC Decoder
        '''
        self.pc_decoder = self.prior_model.module.decoder
        self.pc_decoder = self.model_util.freeze_model(self.model_util.set_model_parallel_gpu(self.pc_decoder))

    def log_step_loss(self, sum_loss, cd_loss, kld_loss, step_idx):
        self.avg_step_loss += sum_loss
        self.avg_epoch_loss += sum_loss
        self.avg_step_cd_loss += cd_loss
        self.avg_epoch_cd_loss += cd_loss
        self.avg_step_kld_loss += kld_loss
        self.avg_epoch_kld_loss += kld_loss

        if step_idx % config.wandb.step_loss_freq == 0:
            self.avg_step_loss /= config.wandb.step_loss_freq
            logging.info('Epoch %d, %d Step, loss = %.6f' % (self._epoch, step_idx, self.avg_step_loss))

            if ((argument.local_rank is not None) and config.cuda.rank[0] == 0) and config.wandb.visual_flag:
                self.visualizer.log_step_loss(step_idx=step_idx, step_loss=self.avg_step_loss,
                                              loss_name='sum')
                self.visualizer.log_step_loss(step_idx=step_idx, step_loss=self.avg_step_cd_loss,
                                              loss_name='cd')
                self.visualizer.log_step_loss(step_idx=step_idx, step_loss=self.avg_step_kld_loss,
                                              loss_name='kld')
            elif (argument.local_rank is None) and config.wandb.visual_flag:
                self.visualizer.log_step_loss(step_idx=step_idx, step_loss=self.avg_step_loss,
                                              loss_name='sum')
                self.visualizer.log_step_loss(step_idx=step_idx, step_loss=self.avg_step_cd_loss,
                                              loss_name='cd')
                self.visualizer.log_step_loss(step_idx=step_idx, step_loss=self.avg_step_kld_loss,
                                              loss_name='kld')

    def log_epoch_loss(self):
        self.avg_epoch_loss /= self.data_length
        self.avg_epoch_cd_loss /= self.data_length
        self.avg_epoch_kld_loss /= self.data_length

        logging.info('Logging Epoch Loss...')
        if ((argument.local_rank is not None) and config.cuda.rank[0] == 0) and config.wandb.visual_flag:
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='sum',
                                           train_epoch_loss=self.avg_epoch_loss)
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='cd',
                                           train_epoch_loss=self.avg_epoch_cd_loss)
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='kld',
                                           train_epoch_loss=self.avg_epoch_kld_loss)
        elif (argument.local_rank is None) and config.wandb.visual_flag:
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='sum',
                                           train_epoch_loss=self.avg_epoch_loss)
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='cd',
                                           train_epoch_loss=self.avg_epoch_cd_loss)
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='kld',
                                           train_epoch_loss=self.avg_epoch_kld_loss)


def trainVAE():
    if (argument.local_rank is not None) and config.cuda.rank[0] == 0:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    elif argument.local_rank is None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    if (argument.local_rank is not None) and config.cuda.rank[0] == 0:
        config.show_config()
    elif argument.local_rank is None:
        config.show_config()

    train_dataset = PCDataset(config=config, split_dataset_type='train')

    if config.cuda.dataparallel_mode == 'Dataparallel':
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=config.network.batch_size,
                                      shuffle=True,
                                      pin_memory=True,
                                      num_workers=15)
        train_session = VAETrainSession(dataloader=train_dataloader)
        train_session.train()

    elif config.cuda.dataparallel_mode == 'DistributedDataParallel':
        torch.cuda.set_device(argument.local_rank)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        num_replicas=config.cuda.world_size[0],
                                                                        rank=config.cuda.rank[0])
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=config.network.batch_size,
                                      shuffle=(train_sampler is None),
                                      pin_memory=False,
                                      sampler=train_sampler,
                                      num_workers=43,
                                      worker_init_fn=np.random.seed(0))
        train_session = VAETrainSession(dataloader=train_dataloader, sampler=train_sampler)
        train_session.train()


if __name__ == '__main__':
    trainVAE()
