import argparse
import logging
import multiprocessing
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from PCAE.config import Config
from PCAE.dataloader import PCDataset
from PCAE.jobs.networks.loss import chamfer_distance_loss, emd_loss
from PCAE.jobs.networks import Network
from PCAE.jobs.networks.models import PointNetAE, LMNetAE, LMImgEncoder
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
parser.add_argument('--train_dataset_size', type=int, required=True,
                    help='The size of train dataset')
parser.add_argument('--test_dataset_size', type=int, required=True,
                    help='The size of test dataset')
parser.add_argument('--valid_dataset_size', type=int, required=True,
                    help='The size of valid dataset')
parser.add_argument('--resample_amount', type=int, required=True, default=2048,
                    help='The num of points to sample from original point cloud')
parser.add_argument('--train_half_class', type=str,
                    help='Train with half of the classes')
'''network
'''
parser.add_argument('--mode_flag', type=str, required=True,
                    help='Mode to train')
parser.add_argument('--img_encoder', type=str,
                    help='Which Image encoder')
parser.add_argument('--prior_model', type=str, required=True,
                    help='Which point cloud autoencoder')
parser.add_argument('--checkpoint_path', type=str, required=True,
                    help='Where to store/load weights')
parser.add_argument('--prior_epoch', type=str, default='300',
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
parser.add_argument('--learning_rate', type=float, required=True,
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


class AEValidSession(Network):
    def __init__(self, dataloader, model=None):
        super().__init__(config=config, data_loader=dataloader, data_type='valid', epoch=1, model=model)
        self._epoch_of_model = ''
        self._epoch = 0

        self.avg_epoch_cd_loss = 0.0
        self.avg_epoch_emd_loss = 0.0
        self.model_util = ModelUtil(config=config)
        self.models_path = self.model_util.get_models_path(config.network.checkpoint_path)
        self.visualizer = WandbVisualizer(config=config, job_type='valid',
                                          model=LMNetAE(config.dataset.resample_amount))

    def validate(self):
        for i, model_path in enumerate(self.models_path):
            self._epoch = self.model_util.get_epoch_num(model_path) - 1
            self.set_model()
            self.model_util.test_trained_model(model=self.model, test_epoch=i + 1)

            self.model.eval()
            with torch.no_grad():
                for idx, (inputs_pc, targets, _) in tqdm(enumerate(self.get_data())):
                    _, predicts = self.model(inputs_pc)
                    cd_loss = chamfer_distance_loss(predicts, targets)
                    _emd_loss = emd_loss(predicts, targets)
                    self.avg_epoch_cd_loss += cd_loss.item()
                    self.avg_epoch_emd_loss += _emd_loss.item()

            self.log_epoch_loss()
            self.avg_epoch_cd_loss = .0
            self.avg_epoch_emd_loss = .0

    def set_model(self):
        models = {'PointNetAE': PointNetAE, 'LMNetAE': LMNetAE, 'LMImgEncoder': LMImgEncoder}
        self.model = models[config.network.prior_model](config.dataset.resample_amount)
        self.model = self.model_util.set_model_device(self.model)
        self.model = self.model_util.set_model_parallel_gpu(self.model)

    def log_epoch_loss(self):
        self.avg_epoch_cd_loss /= config.dataset.dataset_size[self._data_type]
        self.avg_epoch_emd_loss /= config.dataset.dataset_size[self._data_type]

        logging.info('Logging Epoch Loss...')
        if config.wandb.visual_flag:
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_type='cd',
                                           valid_epoch_loss=self.avg_epoch_cd_loss)
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_type='emd',
                                           valid_epoch_loss=self.avg_epoch_emd_loss)


def validAE():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    config.show_config()

    valid_dataset = PCDataset(config=config, split_dataset_type='valid')

    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=config.network.batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=multiprocessing.cpu_count() * 5)
    valid_session = AEValidSession(dataloader=valid_dataloader)
    valid_session.validate()


if __name__ == '__main__':
    validAE()
