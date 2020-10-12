import logging
from tqdm import tqdm
import torch
import numpy as np
import argparse
import multiprocessing
from torch.utils.data import DataLoader

from ..dataloader import PCDataset
from .networks.loss import PCLoss
from .networks import Network
from .networks.models import PointNetAE, LMNetAE, LMImgEncoder
from ..visualizer import WandbVisualizer
from ..config import config
from ..utils import model_util


class ValidSession(Network):
    def __init__(self, dataloader, model=None):
        super().__init__(model=model,
                         data_loader=dataloader,
                         data_type='valid',
                         loss_func=PCLoss().loss,
                         optimizer=config.network.optimizer,
                         epoch=1)
        self._epoch_of_model = ''
        self._epoch = 0

        self.models_path = model_util.get_models_path(config.network.checkpoint_path)
        self.avg_epoch_loss = 0.0
        if config.wandb.visual_flag:
            if config.network.mode_flag == 'ae':
                self.visualizer = WandbVisualizer(job_type='valid', model=LMNetAE(config.dataset.resample_amount))
            elif config.network.mode_flag == 'lm':
                self.visualizer = WandbVisualizer(job_type='valid', model=LMImgEncoder(config.network.latent_size))
        if config.network.mode_flag == 'lm':
            self.prior_model = None

    def validate(self):
        for i, model_path in enumerate(self.models_path):
            self._epoch = model_util.get_epoch_num(model_path) - 1
            self.set_model()
            model_util.test_trained_model(model=self.model, test_epoch=i + 1)

            if config.network.mode_flag == 'ae':
                self.model.eval()
            elif config.network.mode_flag == 'lm':
                self.prior_model.eval()
                self.model.eval()

            with torch.no_grad():
                if config.network.mode_flag == 'ae':
                    for idx, (inputs_pc, targets, _) in tqdm(enumerate(self.get_data())):
                        _, predicts = self.model(inputs_pc)
                        loss = self.loss_func(predicts, targets) * config.network.loss_scale_factor
                        self.avg_epoch_loss += loss.item()

                elif config.network.mode_flag == 'lm':
                    for idx, (inputs_img, inputs_pc, targets, _, _) in tqdm(enumerate(self.get_data())):
                        latent_pc, _ = self.prior_model(inputs_pc)
                        latent_img = self.model(inputs_img)
                        loss = torch.nn.L1Loss()(latent_img, latent_pc) * config.network.loss_scale_factor
                        self.avg_epoch_loss += loss.item()

            self.log_epoch_loss()
            self.avg_epoch_loss = .0
        if config.wandb.visual_flag and config.network.mode_flag == 'ae':
            self.visualizer.log_point_clouds(self.tensor_to_numpy(predicts[-1].view(-1, 3)),
                                             self.tensor_to_numpy(targets[-1].view(-1, 3)))

        if config.cuda.dataparallel_mode == 'DistributedDataParallel':
            model_util.cleanup()

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
            self.prior_model.load_state_dict(state_dict=torch.load(f=prior_model_path))

        self.model = model_util.set_model_device(self.model)
        self.model = model_util.set_model_parallel_gpu(self.model)

    def log_epoch_loss(self):
        # self.avg_epoch_loss /= self.steps_in_an_epoch()
        if config.cuda.dataparallel_mode == 'Dataparallel':
            self.avg_epoch_loss /= PCDataset(self._data_type).__len__()
        elif config.cuda.dataparallel_mode == 'DistributedDataParallel':
            self.avg_epoch_loss /= (PCDataset(self._data_type).__len__() / len(config.cuda.parallel_gpu_ids))

        logging.info('Logging Epoch Loss... %.6f' % self.avg_epoch_loss)
        if config.wandb.visual_flag:
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, valid_epoch_loss=self.avg_epoch_loss)

    @property
    def model_num(self):
        return len(self.models_path)


def validate():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
    config.show_config()

    valid_dataset = PCDataset('valid')

    valid_dataloader = None
    if config.cuda.dataparallel_mode == 'Dataparallel':
        valid_dataloader = DataLoader(dataset=valid_dataset,
                                      batch_size=config.network.batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=multiprocessing.cpu_count() * 2)

    elif config.cuda.dataparallel_mode == 'DistributedDataParallel':
        parser = argparse.ArgumentParser()
        parser.add_argument("--local_rank", type=int)
        args = parser.parse_args()
        torch.cuda.set_device(args.local_rank)

        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
        valid_dataloader = DataLoader(dataset=valid_dataset,
                                      batch_size=config.network.batch_size,
                                      shuffle=False,
                                      pin_memory=True,
                                      sampler=valid_sampler,
                                      num_workers=8,
                                      worker_init_fn=np.random.seed(0))

    valida_session = ValidSession(dataloader=valid_dataloader)
    valida_session.validate()
