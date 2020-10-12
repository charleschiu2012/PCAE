import logging
import os
import json
import numpy as np
import multiprocessing
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from ..dataloader import PCDataset
from .networks.loss import PCLoss
from .networks import Network
from .networks.models import PointNetAE, LMNetAE, LMImgEncoder, LMDecoder
from ..visualizer import WandbVisualizer
from ..config import config
from ..utils import model_util


class TestSession(Network):
    def __init__(self, dataloader, test_epoch, model=None):
        super().__init__(model=model,
                         data_loader=dataloader,
                         data_type='test',
                         loss_func=PCLoss(),
                         optimizer=config.network.optimizer,
                         epoch=1)
        self._epoch = test_epoch

        self.predicts_list = []
        self.targets_list = []
        self.avg_cd_loss = .0
        self.avg_emd_loss = .0
        self.visualizer = None
        if config.network.mode_flag == 'lm':
            self.prior_model = None
            self.decoder = None

    def test(self):
        self.set_model()
        if config.wandb.visual_flag:
            self.visualizer = WandbVisualizer(job_type='test', model=self.model)

        logging.info('Start testing')
        if config.network.mode_flag == 'ae':
            self.model.eval()
        elif config.network.mode_flag == 'lm':
            self.prior_model.eval()
            self.decoder.eval()
            self.model.eval()

        with torch.no_grad():
            if config.network.mode_flag == 'ae':
                for idx, (inputs_pc, targets, _) in tqdm(enumerate(self.get_data())):
                    latent, predicts = self.model(inputs_pc)

                    cd_loss = self.loss_func.chamfer_distance(predicts, targets)
                    emd_loss = self.loss_func.emd_loss(predicts, targets)

                    self.predicts_list.extend(self.tensor_to_numpy(predicts.view(predicts.shape[0], -1, 3)))
                    self.targets_list.extend(self.tensor_to_numpy(targets.view(targets.shape[0], -1, 3)))

                    self.avg_cd_loss += cd_loss.item()
                    self.avg_emd_loss += emd_loss.item()

            elif config.network.mode_flag == 'lm':
                for idx, (inputs_img, inputs_pc, targets, _, _) in tqdm(enumerate(self.get_data())):
                    latent_img = self.model(inputs_img)
                    predicts = self.decoder(latent_img)

                    cd_loss = self.loss_func.chamfer_distance(predicts, targets)
                    emd_loss = self.loss_func.emd_loss(predicts, targets)

                    self.predicts_list.extend(self.tensor_to_numpy(predicts.view(predicts.shape[0], -1, 3)))
                    self.targets_list.extend(self.tensor_to_numpy(targets.view(targets.shape[0], -1, 3)))

                    self.avg_cd_loss += cd_loss.item()
                    self.avg_emd_loss += emd_loss.item()

        self.avg_cd_loss /= self.steps_in_an_epoch()
        self.avg_emd_loss /= self.steps_in_an_epoch()

        self.show_avg_loss()
        if config.wandb.visual_flag:
            self.visualizer.log_point_clouds(self.tensor_to_numpy(predicts[-1].view(-1, 3)),
                                             self.tensor_to_numpy(targets[-1].view(-1, 3)))

        logging.info('Finish testing')
        # if config.wandb.visual_flag:  #TODO
        #     self.save_results()

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

            self.decoder = LMDecoder(config.dataset.resample_amount)
            self.decoder = model_util.set_model_device(self.decoder)
            self.decoder = model_util.set_model_parallel_gpu(self.decoder)
            self.decoder = model_util.load_partial_pretrained_model(self.prior_model, config.network.checkpoint_path,
                                                                    self.decoder, 'decoder')

        self.model = model_util.set_model_device(self.model)
        self.model = model_util.set_model_parallel_gpu(self.model)
        model_util.test_trained_model(model=self.model, test_epoch=self._epoch)

    def show_avg_loss(self):
        print('Average CD testing loss = %.6f' % self.avg_cd_loss)
        print('Average EMD testing loss = %.6f' % self.avg_emd_loss)

    def save_results(self):
        valid_model_path = os.path.join('/home/charles/PCAE-Server/valid_models.json')
        pcd_path = []
        with open(valid_model_path, 'r') as reader:
            jf = json.loads(reader.read())
            keys = jf.keys()
            for pc_class in keys:
                for pc_class_with_id in jf[pc_class]:
                    pcd_path.append(pc_class_with_id)

        for idx, (predict, target) in enumerate(zip(self.predicts_list, self.targets_list)):
            save_path = None
            if config.network.mode_flag == 'ae':
                save_path = os.path.join(config.network.checkpoint_path + '/all_data_ae_result', pcd_path[idx])
            elif config.network.mode_flag == 'lm':
                save_path = os.path.join(config.network.checkpoint_path + '/my_lm_result', pcd_path[idx])
            os.makedirs(save_path, exist_ok=True)
            np.save('{}/predict_{}'.format(save_path, config.network.network_model), predict)
            np.save('{}/target'.format(save_path), target)


def test(trained_epoch: int):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    config.show_config()
    test_dataset = PCDataset('test')

    test_dataloader = None
    if config.cuda.dataparallel_mode == 'Dataparallel':
        test_dataloader = DataLoader(dataset=test_dataset,
                                     batch_size=config.network.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     num_workers=multiprocessing.cpu_count() * 2)

    elif config.cuda.dataparallel_mode == 'DistributedDataParallel':
        parser = argparse.ArgumentParser()
        parser.add_argument("--local_rank", type=int)
        args = parser.parse_args()
        torch.cuda.set_device(args.local_rank)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_dataloader = DataLoader(dataset=test_dataset,
                                     batch_size=config.network.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     sampler=test_sampler,
                                     num_workers=8,
                                     worker_init_fn=np.random.seed(0))

    test_session = TestSession(dataloader=test_dataloader, test_epoch=trained_epoch)

    test_session.test()
