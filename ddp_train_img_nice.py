import argparse
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.distributions as distributions
import torchvision.models as models

from PCAE.config import Config
from PCAE.dataloader import PCDataset
from PCAE.networks import Network
from PCAE.models import LMNetAE, NICE, LMImgEncoder
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
parser.add_argument('--prior_epoch', type=str, required=True, default='LMNetAE/epoch300.pth',
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
parser.add_argument('--nice_epoch', type=str, default='NICE/epoch300.pth',
                    help='Which epoch of NICE to use to ImgEncoder')
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


class ImgNICETrainSession(Network):
    def __init__(self, dataloader, model=None, sampler=None):
        super().__init__(config=config, data_loader=dataloader, data_type='train', epoch=1, model=model)
        self._pretrained_epoch = ''
        self._is_scratch = False

        self.avg_step_loss = .0
        self.avg_epoch_loss = .0
        self.pc_flow = None
        self.prior_model = None
        self.model_util = ModelUtil(config=config)
        self.visualizer = None
        if config.cuda.dataparallel_mode == 'DistributedDataParallel':
            self.sampler = sampler

        self.full_dim = 512
        self.hidden = 5
        if config.nice.latent == 'normal':
            self.prior = torch.distributions.Normal(torch.tensor(0.).cuda(), torch.tensor(1.).cuda())

    def train(self):
        self.set_model()
        if ((argument.local_rank is not None) and config.cuda.rank[0] == 0) and config.wandb.visual_flag:
            self.visualizer = WandbVisualizer(config=config, job_type='train', model=self.model)
        elif (argument.local_rank is None) and config.wandb.visual_flag:
            self.visualizer = WandbVisualizer(config=config, job_type='train', model=self.model)

        self.model.train()
        self.prior_model.eval()
        self.pc_flow.eval()
        for epoch_idx in range(self._epoch - 1, config.network.epoch_num):
            logging.info('Start training epoch %d' % (epoch_idx + 1))

            if config.cuda.dataparallel_mode == 'DistributedDataParallel':
                self.sampler.set_epoch(self._epoch)

            final_step = 0
            for idx, (inputs_img, inputs_pc, targets, _, _) in enumerate(self.get_data()):
                final_step = idx
                self.optimizer.zero_grad()
                latent_imgs = self.model(inputs_img)
                img_log_prob_loss = -self.pc_flow(latent_imgs).mean()

                with torch.no_grad():
                    latent_pcs, _ = self.prior_model(inputs_pc)
                    pc_log_prob_loss = -self.pc_flow(latent_pcs).mean()

                loss = torch.nn.MSELoss()(img_log_prob_loss, pc_log_prob_loss) * config.network.loss_scale_factor

                loss.backward()
                self.optimizer.step()

                self.log_step_loss(loss=loss.item(), step_idx=idx + 1)
                self.avg_step_loss = .0

            logging.info('Epoch %d, %d Step' % (self._epoch, final_step))
            self.save_model()
            self.log_epoch_loss()
            self.avg_epoch_loss = .0
            self._epoch += 1

        if config.cuda.dataparallel_mode == 'DistributedDataParallel':
            self.model_util.cleanup()

    def set_model(self):
        # self.model = LMImgEncoder(latent_size=config.network.latent_size)
        self.model = torch.nn.Sequential(models.resnet50(pretrained=True),
                                         torch.nn.Linear(1000, config.network.latent_size))
        self.model = self.model_util.set_model_device(self.model)
        self.model = self.model_util.set_model_parallel_gpu(self.model)
        self._epoch = self.model_util.load_model_pretrain(self.model, self._pretrained_epoch, self._is_scratch)
        """Prior Model
        """
        self.prior_model = LMNetAE(config.dataset.resample_amount)
        self.prior_model = self.model_util.load_trained_model(self.prior_model, config.network.prior_epoch)
        """NICE
        """
        self.pc_flow = NICE(prior=self.prior, coupling=config.nice.coupling, in_out_dim=self.full_dim,
                            mid_dim=config.nice.mid_dim, hidden=self.hidden, mask_config=config.nice.mask_config)
        self.pc_flow = self.model_util.load_trained_model(self.pc_flow, config.nice.nice_epoch)

    def log_step_loss(self, loss, step_idx):
        self.avg_step_loss += loss
        self.avg_epoch_loss += loss

        if step_idx % config.wandb.step_loss_freq == 0:
            self.avg_step_loss /= config.wandb.step_loss_freq
            logging.info('Epoch %d, %d Step, loss = %.6f' % (self._epoch, step_idx, self.avg_step_loss))

            if ((argument.local_rank is not None) and config.cuda.rank[0] == 0) and config.wandb.visual_flag:
                self.visualizer.log_step_loss(step_idx=step_idx, step_loss=self.avg_step_loss, loss_name='log_prob_mse')
            elif (argument.local_rank is None) and config.wandb.visual_flag:
                self.visualizer.log_step_loss(step_idx=step_idx, step_loss=self.avg_step_loss, loss_name='log_prob_mse')

    def log_epoch_loss(self):
        if config.cuda.dataparallel_mode == 'Dataparallel':
            self.avg_epoch_loss /= config.dataset.dataset_size[self._data_type]
        elif config.cuda.dataparallel_mode == 'DistributedDataParallel':
            self.avg_epoch_loss /= (config.dataset.dataset_size[self._data_type] / len(config.cuda.parallel_gpu_ids))

        logging.info('Logging Epoch Loss...')
        if ((argument.local_rank is not None) and config.cuda.rank[0] == 0) and config.wandb.visual_flag:
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='log_prob_mse',
                                           train_epoch_loss=self.avg_epoch_loss)
        elif (argument.local_rank is None) and config.wandb.visual_flag:
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='log_prob_mse',
                                           train_epoch_loss=self.avg_epoch_loss)


def trainImgNICE():
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
        train_session = ImgNICETrainSession(dataloader=train_dataloader)
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
        train_session = ImgNICETrainSession(dataloader=train_dataloader, sampler=train_sampler)
        train_session.train()


if __name__ == '__main__':
    trainImgNICE()
