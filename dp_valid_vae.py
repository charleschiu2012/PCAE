import argparse
import logging
import multiprocessing
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from PCAE.config import Config
from PCAE.dataloader import PCDataset
from PCAE.jobs.networks.loss import KLDLoss, chamfer_distance_loss, emd_loss
from PCAE.jobs.networks import Network
from PCAE.jobs.networks.models import PointNetAE, LMNetAE, LMImgEncoder, LMDecoder, ImgEncoderVAE
from PCAE.visualizer import WandbVisualizer
from PCAE.utils import ModelUtil

parser = argparse.ArgumentParser()

'''cuda
'''
parser.add_argument('--gpu_usage', type=int, required=True, default=8,
                    help='How many gpu you want use')
parser.add_argument('--dataparallel_mode', type=str, required=True,
                    help='Which mode of dataparallel')
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
parser.add_argument('--mode_flag', type=str, required=True, default='vae',
                    help='Mode to valid')
parser.add_argument('--img_encoder', type=str, required=True, default='LMImgEncoder',
                    help='Which Image encoder')
parser.add_argument('--prior_model', type=str, required=True, default='LMNetAE',
                    help='Which point cloud autoencoder')
parser.add_argument('--checkpoint_path', type=str, required=True,
                    default='/data/LMNet-data/checkpoint/DDP/ImgEncoderVAE',
                    help='Where to store/load weights')
parser.add_argument('--prior_epoch', type=str, required=True, default='300',
                    help='Which epoch of autoencoder to use to ImgEncoder')
parser.add_argument('--loss_scale_factor', type=int, required=True, default=10000,
                    help='Scale your loss')
parser.add_argument('--batch_size', type=int, required=True, default=32,
                    help='Batch size of point cloud or image')
parser.add_argument('--latent_size', type=int, required=True, default=512,
                    help='Size of latent')
parser.add_argument('--vae_z_dim', type=int, required=True, default=512,
                    help='Size of vae latent')
parser.add_argument('--epoch_num', type=int, required=True, default=300,
                    help='How many epoch to valid')
parser.add_argument('--learning_rate', type=float, required=True, default=5e-5,
                    help='Learning rate')
'''wandb
'''
parser.add_argument('--project_name', type=str, required=True, default='PCVAE',
                    help='Project name to log in wandb')
parser.add_argument('--run_name', type=str, required=True, default='ImgEncoderVAE',
                    help='Run name to log under project')
parser.add_argument('--machine_id',
                    help='Which machine')
parser.add_argument('--step_loss_freq', type=int, required=True, default=500,
                    help='How many steps for log step loss one time')
parser.add_argument('--visual_flag', action='store_true',
                    help='Use wandb or not')

argument = parser.parse_args()
config = Config(argument)


class VAEValidSession(Network):
    def __init__(self, dataloader, model=None):
        super().__init__(config=config, data_loader=dataloader, data_type='valid', epoch=1, model=model)
        self._pretrained_epoch = ''
        self._is_scratch = False

        self.avg_epoch_kld_loss = .0
        self.avg_epoch_cd_loss = .0
        self.avg_epoch_emd_loss = .0
        self.prior_model = None
        self.decoder = None
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
            self.prior_model.eval()
            self.decoder.eval()
            with torch.no_grad():
                for idx, (inputs_img, inputs_pc, targets, _, _) in tqdm(enumerate(self.get_data())):
                    latent_pc, _ = self.prior_model(inputs_pc)
                    latent_img, mu, log_var = self.model(inputs_img)
                    reconst_imgs = self.decoder(latent_img)

                    kld_loss = KLDLoss(mu, log_var)
                    cd_loss = chamfer_distance_loss(reconst_imgs, targets)
                    _emd_loss = emd_loss(reconst_imgs, targets)
                    self.avg_epoch_kld_loss += kld_loss.item()
                    self.avg_epoch_cd_loss += cd_loss.item()
                    self.avg_epoch_emd_loss += _emd_loss.item()

                self.log_epoch_loss()
                self.avg_epoch_kld_loss = .0
                self.avg_epoch_cd_loss = .0
                self.avg_epoch_emd_loss = .0

    def set_model(self):
        models = {'PointNetAE': PointNetAE, 'LMNetAE': LMNetAE, 'LMImgEncoder': LMImgEncoder}
        '''Img Encoder
        '''
        self.model = ImgEncoderVAE(latent_size=config.network.latent_size, z_dim=config.network.z_dim)
        self.model = self.model_util.set_model_device(self.model)
        self.model = self.model_util.set_model_parallel_gpu(self.model)
        self._epoch = self.model_util.load_model_pretrain(self.model, self._pretrained_epoch, self._is_scratch)
        '''Prior Model
        '''
        self.prior_model = models[config.network.prior_model](config.dataset.resample_amount)
        self.prior_model = self.model_util.set_model_device(self.prior_model)
        self.prior_model = self.model_util.set_model_parallel_gpu(self.prior_model)
        self.prior_model = self.model_util.load_prior_model(self.prior_model)
        '''PC Decoder
        '''
        self.decoder = LMDecoder(config.dataset.resample_amount)
        self.decoder = self.model_util.set_model_device(self.decoder)
        self.decoder = self.model_util.set_model_parallel_gpu(self.decoder)
        self.decoder = self.model_util.load_partial_pretrained_model(self.prior_model, self.decoder, 'decoder')

    def log_epoch_loss(self):
        self.avg_epoch_kld_loss /= config.dataset.dataset_size[self._data_type]
        self.avg_epoch_cd_loss /= config.dataset.dataset_size[self._data_type]
        self.avg_epoch_emd_loss /= config.dataset.dataset_size[self._data_type]

        logging.info('Logging Epoch Loss...')
        if config.wandb.visual_flag:
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_type='kld',
                                           valid_epoch_loss=self.avg_epoch_kld_loss)
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_type='cd',
                                           valid_epoch_loss=self.avg_epoch_cd_loss)
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_type='emd',
                                           valid_epoch_loss=self.avg_epoch_emd_loss)


def validVAE():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    config.show_config()

    valid_dataset = PCDataset(config=config, split_dataset_type='valid')

    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=config.network.batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=multiprocessing.cpu_count() * 5)
    valid_session = VAEValidSession(dataloader=valid_dataloader)
    valid_session.validate()


if __name__ == '__main__':
    validVAE()
