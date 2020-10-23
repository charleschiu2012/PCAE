import open3d
import os
import argparse

from PCAE.config import Config
from PCAE.jobs import train

parser = argparse.ArgumentParser()

'''cuda
'''
parser.add_argument('--parallel_gpu_ids', type=list, required=True, default=[0, 1, 2, 3, 4, 5, 6, 7],
                    help='Gpu ids you want ues')
parser.add_argument('--dataparallel_mode', type=str, required=True,
                    help='Which mode of dataparallel')
'''dataset
'''
parser.add_argument('--dataset_name', type=str, required=True, default='LMNet_ShapeNet_PC',
                    help='The name of the dataset')
parser.add_argument('--dataset_size', required=True, default={'train': 35022, 'test': 8762, 'valid': 8762},
                    help='The size of each split dataset')
parser.add_argument('--resample_amount', type=int, required=True, default=2048,
                    help='The num of points to sample from original point cloud')
'''network
'''
parser.add_argument('--mode_flag', type=str, required=True, default='ae',
                    help='Mode to train')
parser.add_argument('--img_encoder', type=str, required=True, default='LMImgEncoder',
                    help='Which Image encoder')
parser.add_argument('--prior_model', type=str, required=True, default='LMNetAE',
                    help='Which point cloud autoencoder')
parser.add_argument('--checkpoint_path', type=str, required=True, default='/data/LMNet-data/checkpoint/DDP',
                    help='Where to store/load weights')
parser.add_argument('--prior_epoch', type=str, required=True, default='300',
                    help='Which epoch of autoencoder to use to ImgEncoder')
parser.add_argument('--loss_function', type=str, required=True, default='cd',
                    help='Use what loss function')
parser.add_argument('--loss_scale_factor', type=int, required=True, default=10000,
                    help='Scale your loss')
parser.add_argument('--batch_size', type=int, required=True, default=32,
                    help='Batch size of point cloud or image')
parser.add_argument('--latent_size', type=int, required=True, default=512,
                    help='Size of latent')
parser.add_argument('--z_dim', type=int, required=True, default=512,
                    help='Size of vae latent')  #TODO
parser.add_argument('--epoch_num', type=int, required=True, default=300,
                    help='How many epoch to train')
parser.add_argument('--optimizer', type=str, required=True, default='Adam',
                    help='optimizer')
parser.add_argument('--learning_rate', type=float, required=True, default=5e-5,
                    help='Learning rate')
parser.add_argument('--momentum', type=float, required=True, default=0.9,
                    help='Momentum')
'''nice
'''
parser.add_argument('--nice_batch_size', type=int, required=True, default=200,
                    help='Batch size for NICE')
parser.add_argument('--latent_distribution', type=str, required=True, default='normal',
                    help='Prior distribution for NICE')
parser.add_argument('--mid_dim', type=int, required=True, default=128,
                    help='mid_dim')  #TODO
parser.add_argument('--num_iters', type=int, required=True, default=25000,
                    help='Number of iterations')
parser.add_argument('--num_sample', type=int, required=True, default=64,
                    help='Number of samples')
parser.add_argument('--coupling', type=int, required=True, default=4,
                    help='Number of coupling layers')
parser.add_argument('--mask_config', type=float, required=True, default=1.,
                    help='mask_config')  #TODO
'''wandb
'''
parser.add_argument('--project_name', type=str, required=True, default='PCAE',
                    help='Project name to log in wandb')
parser.add_argument('--run_name', type=str, required=True, default='Autoencoder',
                    help='Run name to log under project')
parser.add_argument('--dir_path', type=str,
                    help='Where to store wandb files')
parser.add_argument('--machine_id',
                    help='Which machine')
parser.add_argument('--step_loss_freq', type=int, required=True, default=500,
                    help='How many steps for log step loss one time')
parser.add_argument('--visual_flag', action='store_true',
                    help='Use wandb or not')

argument = parser.parse_args()
config = Config(argument)

