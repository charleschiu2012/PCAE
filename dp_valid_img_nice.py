import argparse
import logging
import torch
from torch.utils.data import DataLoader
import torch.distributions as distributions

from PCAE.config import Config
from PCAE.dataloader import PCDataset
from PCAE.networks import Network
from PCAE.models import LMNetAE, NICE, ImgNICE, LMImgEncoder
from PCAE.loss import chamfer_distance_loss, emd_loss
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
parser.add_argument('--prior_epoch', type=str, required=True, default='300',
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
parser.add_argument('--nice_lr', type=float, required=True, default=5e-5,
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
parser.add_argument('--nice_epoch', type=str,
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


class ImgNICEValidSession(Network):
    def __init__(self, dataloader, model=None):
        super().__init__(config=config, data_loader=dataloader, data_type='train', epoch=1, model=model)

        self.avg_epoch_cd_loss = .0
        self.avg_epoch_emd_loss = .0
        self.pc_flow = None
        self.img_flow = None
        self.prior_model = None
        self.pc_decoder = None
        self.model_util = ModelUtil(config=config)
        self.models_path = self.model_util.get_models_path(config.network.checkpoint_path)
        self.visualizer = None

        self.full_dim = 512
        self.hidden = 5
        if config.nice.latent == 'normal':
            self.prior = torch.distributions.Normal(torch.tensor(0.).cuda(), torch.tensor(1.).cuda())

        self.visualizer = WandbVisualizer(config=config, job_type='valid',
                                          model=LMImgEncoder(latent_size=config.network.latent_size))

    def validate(self):
        self.set_model()
        for i, model_path in enumerate(self.models_path):
            self._epoch = self.model_util.get_epoch_num(model_path) - 1
            self.model_util.test_trained_model(model=self.model, test_epoch=i + 1)
            self.img_flow = self.model_util.load_trained_model(self.img_flow, "ImgFlow/epoch%.3d.pth" % (i+1))

            self.model.eval()
            self.pc_decoder.eval()
            self.pc_flow.eval()
            self.img_flow.eval()

            final_step = 0
            with torch.no_grad():
                for idx, (inputs_img, inputs_pc, targets, _, _) in enumerate(self.get_data()):
                    final_step = idx

                    latent_imgs = self.model(inputs_img)
                    z, _ = self.img_flow.module.f(latent_imgs)
                    re_latents = self.pc_flow.module.g(z)
                    prediction_imgs = self.pc_decoder(re_latents)

                    cd_loss = chamfer_distance_loss(prediction_imgs, targets)
                    _emd_loss = emd_loss(prediction_imgs, targets)

                    self.avg_epoch_cd_loss += (cd_loss.item() * len(inputs_pc))
                    self.avg_epoch_emd_loss += (_emd_loss.item() * len(inputs_pc))

            logging.info('Epoch %d, %d Step' % (self._epoch, final_step))
            self.log_epoch_loss()
            self.avg_epoch_cd_loss = .0
            self.avg_epoch_emd_loss = .0

    def set_model(self):
        self.model = LMImgEncoder(config.network.latent_size)
        self.model = self.model_util.set_model_device(self.model)
        self.model = self.model_util.set_model_parallel_gpu(self.model)
        """Img Flow
        """
        self.img_flow = ImgNICE(coupling=config.nice.coupling, in_out_dim=self.full_dim,
                                mid_dim=config.nice.mid_dim, hidden=self.hidden, mask_config=config.nice.mask_config)
        self.img_flow = self.model_util.set_model_device(self.img_flow)
        self.img_flow = self.model_util.set_model_parallel_gpu(self.img_flow)
        '''Prior Model & PC Decoder
        '''
        self.prior_model = LMNetAE(config.dataset.resample_amount)
        self.prior_model = self.model_util.set_model_device(self.prior_model)
        self.prior_model = self.model_util.set_model_parallel_gpu(self.prior_model)
        self.prior_model = self.model_util.load_trained_model(self.prior_model, config.network.prior_epoch)
        self.pc_decoder = self.prior_model.module.decoder
        self.pc_decoder = self.model_util.freeze_model(self.model_util.set_model_parallel_gpu(self.pc_decoder))
        '''NICE
        '''
        self.pc_flow = NICE(prior=self.prior, coupling=config.nice.coupling, in_out_dim=self.full_dim,
                            mid_dim=config.nice.mid_dim, hidden=self.hidden, mask_config=config.nice.mask_config)
        self.pc_flow = self.model_util.set_model_device(self.pc_flow)
        self.pc_flow = self.model_util.set_model_parallel_gpu(self.pc_flow)
        self.pc_flow = self.model_util.load_trained_model(self.pc_flow, config.nice.nice_epoch)

    def log_epoch_loss(self):
        self.avg_epoch_cd_loss /= config.dataset.dataset_size[self._data_type]
        self.avg_epoch_emd_loss /= config.dataset.dataset_size[self._data_type]

        logging.info('Logging Epoch Loss...')
        self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='cd',
                                       valid_epoch_loss=self.avg_epoch_cd_loss)
        self.visualizer.log_epoch_loss(epoch_idx=self._epoch, loss_name='emd',
                                       valid_epoch_loss=self.avg_epoch_emd_loss)


def validImgNICE():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    config.show_config()

    valid_dataset = PCDataset(config=config, split_dataset_type='train')

    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=config.network.batch_size,
                                  shuffle=False,
                                  pin_memory=False,
                                  num_workers=43)
    train_session = ImgNICEValidSession(dataloader=valid_dataloader)
    train_session.validate()


if __name__ == '__main__':
    validImgNICE()
