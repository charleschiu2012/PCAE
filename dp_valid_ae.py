import argparse
import logging
import torch
from torch.utils.data import DataLoader

from PCAE.config import Config
from PCAE.dataloader import PCDataset, ModelNet10
from PCAE.loss import chamfer_distance_loss, emd_loss
from PCAE.networks import Network
from PCAE.models import PointNetAE, LMNetAE, LMImgEncoder
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
parser.add_argument('--resample_amount', type=int, required=True,
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
parser.add_argument('--prior_model', type=str, required=True,
                    help='Which point cloud autoencoder')
parser.add_argument('--checkpoint_path', type=str, required=True,
                    help='Where to store/load weights')
parser.add_argument('--prior_epoch', type=str, default='300',
                    help='Which epoch of autoencoder to use to ImgEncoder')
parser.add_argument('--img_encoder_epoch', type=str,
                    help='Which epoch of ImgEncoder')
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
parser.add_argument('--nice_epoch', type=str,
                    help='which epoch of NICE')
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

        self.avg_epoch_cd_loss = 0.0
        self.avg_epoch_emd_loss = 0.0
        self.data_length = 0
        self.model_util = ModelUtil(config=config)
        self.models_path = self.model_util.get_models_path(config.network.checkpoint_path)
        self.visualizer = WandbVisualizer(config=config, job_type='valid',
                                          model=LMNetAE(config.dataset.resample_amount, config.network.latent_size))

    def validate(self):
        self.set_model()
        for i, model_path in enumerate(self.models_path):
            self._epoch = self.model_util.get_epoch_num(model_path) - 1
            self.model_util.test_trained_model(model=self.model, test_epoch=i + 1)

            self.model.eval()
            final_step = 0
            with torch.no_grad():
                for idx, (inputs_pc, targets, _) in enumerate(self.get_data()):
                    final_step = idx
                    self.data_length += len(inputs_pc)

                    _, predicts = self.model(inputs_pc)
                    cd_loss = chamfer_distance_loss(predicts, targets)
                    # _emd_loss = emd_loss(predicts, targets)
                    self.avg_epoch_cd_loss += (cd_loss.item() * len(inputs_pc))
                    # self.avg_epoch_emd_loss += (_emd_loss.item() * len(inputs_pc))

            logging.info('Epoch %d, %d Step' % (self._epoch, final_step))
            self.log_epoch_loss()
            self.avg_epoch_cd_loss = .0
            # self.avg_epoch_emd_loss = .0
            self.data_length = 0

    def set_model(self):
        models = {'PointNetAE': PointNetAE, 'LMNetAE': LMNetAE, 'LMImgEncoder': LMImgEncoder}
        self.model = models[config.network.prior_model](config.dataset.resample_amount, config.network.latent_size)
        self.model = self.model_util.set_model_device(self.model)
        self.model = self.model_util.set_model_parallel_gpu(self.model)

    def log_epoch_loss(self):
        self.avg_epoch_cd_loss /= self.data_length
        # self.avg_epoch_emd_loss /= self.data_length

        logging.info('Logging Epoch Loss...')
        if config.wandb.visual_flag:
            if not config.dataset.test_unseen_flag:
                self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='cd',
                                               valid_epoch_loss=self.avg_epoch_cd_loss)
                # self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='emd',
                #                                valid_epoch_loss=self.avg_epoch_emd_loss)
            if config.dataset.test_unseen_flag:
                self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='unseen_cd',
                                               valid_epoch_loss=self.avg_epoch_cd_loss)
                # self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='unseen_emd',
                #                                valid_epoch_loss=self.avg_epoch_emd_loss)


def validAE():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    config.show_config()

    valid_dataset = None
    if argument.dataset_name == 'LMNet_ShapeNet_PC':
        valid_dataset = PCDataset(config=config, split_dataset_type='valid')
    elif argument.dataset_name == 'ModelNet10':
        root_dir = '/home/justice113/data/modelnet10_hdf5_2048/'
        valid_dataset = ModelNet10(root_dir=root_dir, subset='test',
                                   num_max=2048, num_sample=config.dataset.resample_amount)

    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=config.network.batch_size,
                                  shuffle=False,
                                  pin_memory=False,
                                  num_workers=22)
    valid_session = AEValidSession(dataloader=valid_dataloader)
    valid_session.validate()


if __name__ == '__main__':
    validAE()
