import logging
import argparse
import multiprocessing
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.distributions as distributions

from ..dataloader import FlowDataset
from .networks import Network
from .networks.models import NICE, LMNetAE, LMDecoder
from ..visualizer import WandbVisualizer, tsne
from ..config import config
from ..utils import model_util


class ValidNICESession(Network):
    def __init__(self, dataloader, model=None):
        super().__init__(model=model, data_loader=dataloader, data_type='train',
                         loss_func="None", optimizer=config.network.optimizer, epoch=1)

        self._epoch_of_model = ''
        self._epoch = 0

        self.models_path = model_util.get_models_path(config.network.checkpoint_path)
        self.latent_list = []
        self.label_list = []
        self.latents = []
        self.latent_ids = []
        self.visualizer = None
        self.prior_model = None
        self.decoder = None

        self.flow = None
        self.full_dim = 512
        self.hidden = 5
        if config.nice.latent == 'normal':
            self.prior = torch.distributions.Normal(torch.tensor(0.).cuda(), torch.tensor(1.).cuda())
        if config.wandb.visual_flag:
            self.visualizer = WandbVisualizer(job_type='valid', model=NICE(prior=self.prior,
                                                                           coupling=config.nice.coupling,
                                                                           in_out_dim=self.full_dim,
                                                                           mid_dim=config.nice.mid_dim,
                                                                           hidden=self.hidden,
                                                                           mask_config=config.nice.mask_config))

    def valid(self):
        #         for i, model_path in enumerate(self.models_path):
        #         self._epoch = model_util.get_epoch_num(model_path) - 1
        self._epoch = 300
        self.set_model()
        #         model_util.test_trained_model(model=self.model, test_epoch=i + 1)
        model_util.test_trained_model(model=self.flow, test_epoch=self._epoch)
        self.flow.eval()
        self.prior_model.eval()

        with torch.no_grad():
            for idx, (ae_latents, latent_ids) in tqdm(enumerate(self.get_data())):
                #  reconstructions
                z, _ = self.flow.module.f(ae_latents)
                reconst_latents = self.flow.module.g(z)
                # #                     reconst = utils.prepare_data(
                # #                         reconst, dataset, zca=zca, mean=mean, reverse=True)
                reconst_pcs = self.decoder(ae_latents)
                flow_pcs = self.decoder(reconst_latents)
                # for idx in range(len(ae_latents)):
                #     fig = plt.figure(figsize=(10, 7))
                #     print(flow_pcs[idx])
                #     plot_pcd(121, fig, flow_pcs[idx].detach().cpu().numpy(), 'Flow: \n' + latent_ids[idx])
                #     print(reconst_pcs[idx])
                #     plot_pcd(122, fig, reconst_pcs[idx].detach().cpu().numpy(), 'AE: \n' + latent_ids[idx])
                #
                #     plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
                #     plt.show()
                #     break
                # break
            #  samples
            samples = self.flow.module.sample(config.nice.sample_size)
            # samples = utils.prepare_data(
            #     samples, dataset, zca=zca, mean=mean, reverse=True)
            # loss = -self.model(ae_latents).mean()
            sample_pcs = self.decoder(samples)
            # for idx in range(config.nice.sample_size):
            #     fig = plt.figure(figsize=(10, 3))
            #     plot_pcd(111, fig, sample_pcs[idx].detach().cpu().numpy(), 'Sample: ')
            #
            #     plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
            #     plt.show()

    def set_model(self):
        self.flow = NICE(prior=self.prior,
                         coupling=config.nice.coupling,
                         in_out_dim=self.full_dim,
                         mid_dim=config.nice.mid_dim,
                         hidden=self.hidden,
                         mask_config=config.nice.mask_config)
        self.flow = model_util.set_model_device(self.flow)
        self.flow = model_util.set_model_parallel_gpu(self.flow)
        #         print(self.flow)

        self.prior_model = LMNetAE(config.dataset.resample_amount)
        prior_model_path = '%s/%s/epoch%.3d.pth' % (config.network.checkpoint_path,
                                                    config.network.prior_model,
                                                    int(config.network.prior_epoch))
        self.prior_model = model_util.set_model_device(self.prior_model)
        self.prior_model = model_util.set_model_parallel_gpu(self.prior_model)
        self.prior_model.load_state_dict(state_dict=torch.load(f=prior_model_path))
        #         print(self.prior_model)

        self.decoder = LMDecoder(config.dataset.resample_amount)
        self.decoder = model_util.set_model_device(self.decoder)
        self.decoder = model_util.set_model_parallel_gpu(self.decoder)
        self.decoder = model_util.load_partial_pretrained_model(self.prior_model, config.network.prior_epoch,
                                                                self.decoder, 'decoder')

    #         print(self.decoder)

    def log_epoch_loss(self):
        if config.cuda.dataparallel_mode == 'Dataparallel':
            self.avg_epoch_loss /= FlowDataset(self._data_type).__len__()
        elif config.cuda.dataparallel_mode == 'DistributedDataParallel':
            self.avg_epoch_loss /= (FlowDataset(self._data_type).__len__() / len(config.cuda.parallel_gpu_ids))

        logging.info('Logging Epoch Loss...')
        if config.wandb.visual_flag:
            self.visualizer.log_epoch_loss(epoch_idx=self._epoch, train_epoch_loss=self.avg_epoch_loss)
            self.avg_epoch_loss = .0


def valid_nice():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    #     config.show_config()

    valid_dataset = FlowDataset('train')

    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=config.network.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=multiprocessing.cpu_count() * 5)
    valid_session = ValidNICESession(dataloader=valid_dataloader)
    valid_session.valid()
