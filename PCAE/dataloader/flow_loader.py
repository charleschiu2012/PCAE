import os
import json
import numpy as np
import copy
import itertools
import random
from torchvision import transforms
from torch.utils.data import Dataset
from ..config import config
from ..utils import pointcloud_util, shapenet_taxonomy


class FlowDataset(Dataset):
    def __init__(self, split_dataset_type: str):
        self.split_dataset_type = split_dataset_type
        self.dataset_loader = FlowLoader(split_dataset_type)

    def __len__(self):
        return len(self.dataset_loader.ae_latent_paths)

    def __getitem__(self, item):
        ae_latent, ae_id = self.get_ae_latent(item)
        lm_latent, lm_id = self.get_lm_latent(item)

        assert ae_id == lm_id
        return ae_latent, lm_latent, ae_id

    def get_ae_latent(self, item):
        ae_latent_path = self.dataset_loader.ae_latent_paths[item]
        ae_latent = np.load(ae_latent_path)
        ae_id = '/'.join(ae_latent_path.split('/')[8:10])

        return ae_latent, ae_id

    def get_lm_latent(self, item):
        lm_latent_pc_id_paths = self.dataset_loader.lm_latent_paths[item]
        chosen_view = random.randint(0, 24)
        lm_latent_path = lm_latent_pc_id_paths[chosen_view]
        lm_latent = np.load(lm_latent_path)
        lm_id = '/'.join(lm_latent_path.split('/')[8:10])

        return lm_latent, lm_id


class FlowLoader:
    def __init__(self, split_dataset_type: str):
        assert split_dataset_type in ['train', 'test', 'valid']

        self.split_dataset_type = split_dataset_type
        self.split_dataset_path = None
        self.pc_ids = []
        self.ae_latent_paths = []
        self.lm_latent_paths = []

        self.set_split_dataset_path()
        self.load_pc_ids()
        self.load_ae_latents()
        self.load_lm_latents()

    def set_split_dataset_path(self):
        if self.split_dataset_type == 'test':
            self.split_dataset_path = os.path.join(config.dataset.dataset_path,
                                                   'valid_models.json')
        else:
            self.split_dataset_path = os.path.join(config.dataset.dataset_path,
                                                   str(self.split_dataset_type) + '_models.json')

    def load_pc_ids(self):
        with open(self.split_dataset_path, 'r') as reader:
            jf = json.loads(reader.read())

            train_keys = [shapenet_taxonomy.shapenet_category_to_id[class_id]
                          for class_id in config.flow.train_class]

            for pc_class in jf.keys():
                for train_key in train_keys:
                    if pc_class == train_key:
                        for pc_class_with_id in jf[pc_class]:
                            self.pc_ids.append(pc_class_with_id)

            if config.dataset.get_dataset_num(self.split_dataset_type) < len(self.pc_ids):
                random.shuffle(self.pc_ids)
                self.pc_ids = self.pc_ids[:config.flow.ae_dataset_size[self.split_dataset_type]]

    def load_ae_latents(self):
        #  self.lm_latent_paths shape = [dim_0 = pc_id]
        for pc_id in self.pc_ids:
            self.ae_latent_paths.append(os.path.join(config.network.checkpoint_path +
                                                     'train_ae_latent/', pc_id, 'latent.npy'))

    def load_lm_latents(self):
        #  self.ae_latent_paths shape = [dim_0 = pc_id, dim_1= num_views]
        for pc_id in self.pc_ids:
            id_level = []
            self.ae_latent_paths.append(id_level)
            for idx in range(24):
                self.ae_latent_paths[-1].append(os.path.join(config.network.checkpoint_path +
                                                         'train_lm_latent/', pc_id, str(idx).zfill(2), 'latent.npy'))
