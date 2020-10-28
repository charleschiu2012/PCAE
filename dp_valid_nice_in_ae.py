import argparse
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.distributions as distributions

from PCAE.config import Config
from PCAE.dataloader import PCDataset
from PCAE.jobs.networks import Network
from PCAE.jobs.networks.models import NiceAE
from PCAE.jobs.networks.loss import chamfer_distance_loss
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
parser.add_argument('--img_encoder', type=str,
                    help='Which Image encoder')
parser.add_argument('--prior_model', type=str,
                    help='Which point cloud autoencoder')
parser.add_argument('--checkpoint_path', type=str, required=True,
                    help='Where to store/load weights')
parser.add_argument('--prior_epoch', type=str,
                    help='Which epoch of autoencoder to use to ImgEncoder')
parser.add_argument('--loss_scale_factor', type=int, required=True, default=10000,
                    help='Scale your loss')
parser.add_argument('--batch_size', type=int, required=True, default=32,
                    help='Batch size of point cloud or image')
parser.add_argument('--latent_size', type=int, required=True, default=512,
                    help='Size of latent')
parser.add_argument('--z_dim', type=int, default=512,
                    help='Size of vae latent')
parser.add_argument('--epoch_num', type=int, required=True, default=300,
                    help='How many epoch to train')
parser.add_argument('--learning_rate', type=float, required=True, default=5e-5,
                    help='Learning rate')
'''nice
'''
parser.add_argument('--nice_batch_size', type=int, required=True, default=200,
                    help='Batch size for NICE')
parser.add_argument('--latent_distribution', type=str, required=True, default='normal',
                    help='Prior distribution for NICE')
parser.add_argument('--mid_dim', type=int, required=True, default=128,
                    help='mid_dim')  # TODO
parser.add_argument('--num_iters', type=int, required=True, default=25000,
                    help='Number of iterations')
parser.add_argument('--num_sample', type=int, required=True, default=64,
                    help='Number of samples')
parser.add_argument('--coupling', type=int, required=True, default=4,
                    help='Number of coupling layers')
parser.add_argument('--mask_config', type=float, required=True, default=1.,
                    help='mask_config')  # TODO
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


class NICEAEValidSession(Network):
    def __init__(self, dataloader, model=None):
        super().__init__(config=config, data_loader=dataloader, data_type='valid', epoch=1, model=model)
        self._epoch_of_model = ''
        self._epoch = 0

        self.avg_epoch_loss = 0.0
        self.avg_epoch_cd_loss = 0.0
        self.avg_epoch_log_prob_loss = 0.0
        self.prior_model = None
        self.decoder = None
        self.model_util = ModelUtil(config=config)
        self.models_path = self.model_util.get_models_path(config.network.checkpoint_path)

        self.full_dim = 512
        self.hidden = 5
        if config.nice.latent == 'normal':
            self.prior = torch.distributions.Normal(torch.tensor(0.).cuda(), torch.tensor(1.).cuda())
        self.visualizer = WandbVisualizer(config=config, job_type='valid',
                                          model=NiceAE(prior=self.prior,
                                                       coupling=config.nice.coupling,
                                                       in_out_dim=self.full_dim,
                                                       mid_dim=config.nice.mid_dim,
                                                       hidden=self.hidden,
                                                       mask_config=config.nice.mask_config,
                                                       num_points=config.dataset.resample_amount))

    def validate(self):
        for i, model_path in enumerate(self.models_path):
            self._epoch = self.model_util.get_epoch_num(model_path) - 1
            self.set_model()
            self.model_util.test_trained_model(model=self.model, test_epoch=i + 1)

            self.model.eval()
            with torch.no_grad():
                final_step = 0
                for idx, (inputs_pc, targets, pc_ids) in enumerate(self.get_data()):
                    final_step = idx
                    log_prob_loss, predictions = self.model(inputs_pc)

                    cd_loss = chamfer_distance_loss(predictions, targets) * config.network.loss_scale_factor
                    log_prob_loss = log_prob_loss.mean()
                    loss = log_prob_loss + cd_loss
                    self.avg_epoch_loss += loss.item()
                    self.avg_epoch_cd_loss += cd_loss.item()
                    self.avg_epoch_log_prob_loss += log_prob_loss.item()

                logging.info('Epoch %d, %d Step' % (self._epoch, final_step))
                self.log_epoch_loss()
                self.avg_epoch_loss = .0
                self.avg_epoch_cd_loss = .0
                self.avg_epoch_log_prob_loss = .0

    def set_model(self):
        self.model = NiceAE(prior=self.prior,
                            coupling=config.nice.coupling,
                            in_out_dim=self.full_dim,
                            mid_dim=config.nice.mid_dim,
                            hidden=self.hidden,
                            mask_config=config.nice.mask_config,
                            num_points=config.dataset.resample_amount)
        self.model = self.model_util.set_model_device(self.model)
        self.model = self.model_util.set_model_parallel_gpu(self.model)

    def log_epoch_loss(self):
        self.avg_epoch_loss /= config.dataset.dataset_size[self._data_type]
        self.avg_epoch_cd_loss /= config.dataset.dataset_size[self._data_type]
        self.avg_epoch_log_prob_loss /= config.dataset.dataset_size[self._data_type]

        logging.info('Logging Epoch Loss...')
        if ((argument.local_rank is not None) and config.cuda.rank[0] == 0) and config.wandb.visual_flag:
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='sum',
                                           valid_epoch_loss=self.avg_epoch_loss)
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='cd',
                                           valid_epoch_loss=self.avg_epoch_cd_loss)
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='log_prob',
                                           valid_epoch_loss=self.avg_epoch_log_prob_loss)

        elif (argument.local_rank is None) and config.wandb.visual_flag:
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='sum',
                                           valid_epoch_loss=self.avg_epoch_loss)
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='cd',
                                           valid_epoch_loss=self.avg_epoch_cd_loss)
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='log_prob',
                                           valid_epoch_loss=self.avg_epoch_log_prob_loss)


def validateNICEAE():
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

    valid_dataset = PCDataset(config=config, split_dataset_type='valid')

    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=config.network.batch_size,
                                  shuffle=False,
                                  pin_memory=False,
                                  num_workers=22)
    valid_session = NICEAEValidSession(dataloader=valid_dataloader)
    valid_session.validate()


if __name__ == '__main__':
    validateNICEAE()
