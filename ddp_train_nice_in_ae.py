import argparse
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.distributions as distributions

from PCAE.config import Config
from PCAE.dataloader import PCDataset
from PCAE.networks import Network
from PCAE.models import NICE, LMNetAE
from PCAE.loss import chamfer_distance_loss
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
parser.add_argument('--nice_epoch', type=str, default=None)
'''nice
'''
parser.add_argument('--nice_batch_size', type=int, required=True, default=200,
                    help='Batch size for NICE')
parser.add_argument('--nice_lr', type=float, required=True,
                    help='learning rate for NICE flow')
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


class NICEAETrainSession(Network):
    def __init__(self, dataloader, model=None, sampler=None):
        super().__init__(config=config, data_loader=dataloader, data_type='train', epoch=1, model=model)
        self._pretrained_epoch = ''
        self._is_scratch = False

        self.avg_step_loss = 0.0
        self.avg_epoch_loss = 0.0
        self.avg_step_cd_loss = 0.0
        self.avg_epoch_cd_loss = 0.0
        self.avg_step_log_prob_loss = 0.0
        self.avg_epoch_log_prob_loss = 0.0
        self.prior_model = None
        self.decoder = None
        self.pc_flow = None
        self.model_util = ModelUtil(config=config)
        self.visualizer = None
        if config.cuda.dataparallel_mode == 'DistributedDataParallel':
            self.sampler = sampler

        self.full_dim = 512
        self.hidden = 5
        if config.nice.latent == 'normal':
            self.prior = torch.distributions.Normal(torch.tensor(0.).cuda(), torch.tensor(1.).cuda())
        self.optimizer_f = None

    def train(self):
        self.set_model()
        if ((argument.local_rank is not None) and config.cuda.rank[0] == 0) and config.wandb.visual_flag:
            self.visualizer = WandbVisualizer(config=config, job_type='train', model=self.model)
        elif (argument.local_rank is None) and config.wandb.visual_flag:
            self.visualizer = WandbVisualizer(config=config, job_type='train', model=self.model)

        self.model.train()
        for epoch_idx in range(self._epoch - 1, config.network.epoch_num):
            logging.info('Start training epoch %d' % (epoch_idx + 1))

            if config.cuda.dataparallel_mode == 'DistributedDataParallel':
                self.sampler.set_epoch(self._epoch)

            final_step = 0
            for idx, (inputs_pc, targets, pc_ids) in enumerate(self.get_data()):
                final_step = idx
                self.optimizer.zero_grad()
                self.optimizer_f.zero_grad()
                latent_pcs, predictions = self.model(inputs_pc)

                log_prob_loss = -self.pc_flow(x=latent_pcs).mean()
                cd_loss = chamfer_distance_loss(predictions, targets) * config.network.loss_scale_factor
                loss = cd_loss + log_prob_loss

                loss.backward()
                self.optimizer.step()
                self.optimizer_f.step()
                true_cd_loss = cd_loss / config.network.loss_scale_factor
                self.log_step_loss(sum_loss=((true_cd_loss+log_prob_loss).item() * len(inputs_pc)),
                                   cd_loss=(true_cd_loss.item() * len(inputs_pc)),
                                   log_prob_loss=(log_prob_loss.item() * len(inputs_pc)),
                                   step_idx=idx + 1)
                self.avg_step_loss = .0
                self.avg_step_cd_loss = .0
                self.avg_step_log_prob_loss = .0

            logging.info('Epoch %d, %d Step' % (self._epoch, final_step))
            ae_nice_ck_path = config.network.checkpoint_path + '/AE'
            self.model_util.save_model(model=self.model, ck_path=ae_nice_ck_path, epoch=self._epoch)
            pc_flow_ck_path = config.network.checkpoint_path + '/PC_Flow'
            self.model_util.save_model(model=self.pc_flow, ck_path=pc_flow_ck_path, epoch=self._epoch)
            self.log_epoch_loss()
            self.avg_epoch_loss = .0
            self.avg_epoch_cd_loss = .0
            self.avg_epoch_log_prob_loss = .0
            self._epoch += 1

        if config.cuda.dataparallel_mode == 'DistributedDataParallel':
            self.model_util.cleanup()

    def set_model(self):
        """Prior Model
        """
        self.model = LMNetAE(config.dataset.resample_amount)
        self.model = self.model_util.set_model_device(self.model)
        self.model = self.model_util.set_model_parallel_gpu(self.model)
        self._epoch = self.model_util.load_model_pretrain(self.model, self._pretrained_epoch, self._is_scratch)
        """PC flow
        """
        self.pc_flow = NICE(prior=self.prior, coupling=config.nice.coupling, in_out_dim=self.full_dim,
                            mid_dim=config.nice.mid_dim, hidden=self.hidden, mask_config=config.nice.mask_config)
        self.pc_flow = self.model_util.set_model_device(self.pc_flow)
        self.pc_flow = self.model_util.set_model_parallel_gpu(self.pc_flow)
        self.optimizer_f = torch.optim.Adam(params=self.pc_flow.parameters(), lr=config.nice.learning_rate,
                                            betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    def log_step_loss(self, sum_loss, cd_loss, log_prob_loss, step_idx):
        self.avg_step_loss += sum_loss
        self.avg_epoch_loss += sum_loss
        self.avg_step_cd_loss += cd_loss
        self.avg_epoch_cd_loss += cd_loss
        self.avg_step_log_prob_loss += log_prob_loss
        self.avg_epoch_log_prob_loss += log_prob_loss

        if step_idx % config.wandb.step_loss_freq == 0:
            self.avg_step_loss /= config.wandb.step_loss_freq
            self.avg_step_cd_loss /= config.wandb.step_loss_freq
            self.avg_step_log_prob_loss /= config.wandb.step_loss_freq
            logging.info('Epoch %d, %d Step, loss = %.6f' % (self._epoch, step_idx, self.avg_step_loss))

            if ((argument.local_rank is not None) and config.cuda.rank[0] == 0) and config.wandb.visual_flag:
                self.visualizer.log_step_loss(step_idx=step_idx, step_loss=self.avg_step_loss,
                                              loss_name='sum')
                self.visualizer.log_step_loss(step_idx=step_idx, step_loss=self.avg_step_cd_loss,
                                              loss_name='cd')
                self.visualizer.log_step_loss(step_idx=step_idx, step_loss=self.avg_step_log_prob_loss,
                                              loss_name='log_prob')
            elif (argument.local_rank is None) and config.wandb.visual_flag:
                self.visualizer.log_step_loss(step_idx=step_idx, step_loss=self.avg_step_loss,
                                              loss_name='sum')
                self.visualizer.log_step_loss(step_idx=step_idx, step_loss=self.avg_step_cd_loss,
                                              loss_name='cd')
                self.visualizer.log_step_loss(step_idx=step_idx, step_loss=self.avg_step_log_prob_loss,
                                              loss_name='log_prob')

    def log_epoch_loss(self):
        if config.cuda.dataparallel_mode == 'Dataparallel':
            self.avg_epoch_loss /= config.dataset.dataset_size[self._data_type]
            self.avg_epoch_cd_loss /= config.dataset.dataset_size[self._data_type]
            self.avg_epoch_log_prob_loss /= config.dataset.dataset_size[self._data_type]
        elif config.cuda.dataparallel_mode == 'DistributedDataParallel':
            self.avg_epoch_loss /= (config.dataset.dataset_size[self._data_type] / len(config.cuda.parallel_gpu_ids))
            self.avg_epoch_cd_loss /= (config.dataset.dataset_size[self._data_type] / len(config.cuda.parallel_gpu_ids))
            self.avg_epoch_log_prob_loss /= (config.dataset.dataset_size[self._data_type] /
                                             len(config.cuda.parallel_gpu_ids))

        logging.info('Logging Epoch Loss...')
        if ((argument.local_rank is not None) and config.cuda.rank[0] == 0) and config.wandb.visual_flag:
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='sum',
                                           train_epoch_loss=self.avg_epoch_loss)
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='cd',
                                           train_epoch_loss=self.avg_epoch_cd_loss)
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='log_prob',
                                           train_epoch_loss=self.avg_epoch_log_prob_loss)
        elif (argument.local_rank is None) and config.wandb.visual_flag:
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='sum',
                                           train_epoch_loss=self.avg_epoch_loss)
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='cd',
                                           train_epoch_loss=self.avg_epoch_cd_loss)
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='log_prob',
                                           train_epoch_loss=self.avg_epoch_log_prob_loss)


def trainNICEAE():
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
                                      pin_memory=False,
                                      num_workers=22)
        train_session = NICEAETrainSession(dataloader=train_dataloader)
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
                                      num_workers=12,
                                      worker_init_fn=np.random.seed(0))
        train_session = NICEAETrainSession(dataloader=train_dataloader, sampler=train_sampler)
        train_session.train()


if __name__ == '__main__':
    trainNICEAE()
