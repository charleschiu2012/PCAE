import os
import json
import numpy as np
import copy
import itertools
import cv2
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from ..utils import pointcloud_util, shapenet_taxonomy


class PCDataset(Dataset):
    def __init__(self, config, split_dataset_type: str):
        self.config = config
        self.split_dataset_type = split_dataset_type
        self.dataset_loader = DatasetLoader(config=config, split_dataset_type=split_dataset_type)

    def __len__(self):
        # assert len(self.dataset_loader.split_render_dataset_path) == \
        #        config.dataset.get_dataset_num(self.split_dataset_type)

        if (self.config.network.mode_flag == 'ae') or (self.config.network.mode_flag == 'nice'):
            return len(self.dataset_loader.pc_paths)
        elif (self.config.network.mode_flag == 'lm') \
                or (self.config.network.mode_flag == 'vae') \
                or (self.config.network.mode_flag == 'all_view') \
                or (self.config.network.mode_flag == 'img_nice'):
            return len(self.dataset_loader.split_render_dataset_path)

    def __getitem__(self, item):
        if (self.config.network.mode_flag == 'ae') or (self.config.network.mode_flag == 'nice'):
            pc, pc_id = self.get_pc(item)
            target = copy.deepcopy(pc)

            return pc, target, pc_id
        elif (self.config.network.mode_flag == 'lm') or (self.config.network.mode_flag == 'vae') \
                or (self.config.network.mode_flag == 'img_nice'):
            img, img_id = self.get_img(item)
            pc, pc_id = self.get_pc(item)
            target = copy.deepcopy(pc)

            return img, pc, target, img_id, pc_id
        elif self.config.network.mode_flag == 'all_view':
            imgs, img_ids = self.get_all_view_img(item)
            pc, pc_id = self.get_pc(item)
            target = copy.deepcopy(pc)

            return imgs, pc, target, img_ids, pc_id

    def get_pc(self, item):
        pc_path = None
        if (self.config.network.mode_flag == 'ae') or (self.config.network.mode_flag == 'nice'):
            pc_path = self.dataset_loader.pc_paths[item]
        elif (self.config.network.mode_flag == 'lm') \
                or (self.config.network.mode_flag == 'vae') \
                or (self.config.network.mode_flag == 'all_view') \
                or (self.config.network.mode_flag == 'img_nice'):
            pc_path = self.dataset_loader.split_render_dataset_path[item][0]
        # pc = o3d.io.read_point_cloud(pc_path)
        # assert isinstance(pc, o3d.geometry.PointCloud)
        pc = np.load(pc_path)  # N*3
        normalized_pc = pointcloud_util.normalize_pcd(pc)
        resampled_pc = pointcloud_util.resample_pcd(normalized_pc, self.config.dataset.resample_amount)
        # resampled_pc = normalized_pc
        assert isinstance(resampled_pc, np.ndarray)
        assert pointcloud_util.get_point_amount(resampled_pc) > 0

        pc_id = '/'.join(pc_path.split('/')[6:8])

        return resampled_pc, pc_id

    def get_img(self, item):
        img_path = self.dataset_loader.split_render_dataset_path[item][1]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)[4:-5, 4:-5, :3]
        # img = Image.open(img_path).convert('RGB')

        img_id = '/'.join(img_path.split('/')[6:10]).split('.')[0]

        # assert isinstance(img, np.ndarray)
        # assert img.shape[0] > 0

        # transform = transforms.Compose([
        #     transforms.Resize(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])
        #
        # img = transform(img)

        return img, img_id

    def get_all_view_img(self, item):
        img_paths = self.dataset_loader.split_render_dataset_path[item][1]  # 24 paths
        imgs = []
        img_ids = []
        for idx in range(24):
            img = cv2.imread(img_paths[idx], cv2.IMREAD_COLOR)[4:-5, 4:-5, :3]
            # img = Image.open(img_path).convert('RGB')
            imgs.append(img)

            img_id = '/'.join(img_paths[idx].split('/')[6:10]).split('.')[0]
            img_ids.append(img_id)
        imgs = np.array(imgs)

        return imgs, img_ids


class DatasetLoader:
    def __init__(self, config, split_dataset_type: str):
        self.config = config
        assert split_dataset_type in ['train', 'test', 'valid']

        self.train_classes = None
        self.split_dataset_type = split_dataset_type
        self.split_dataset_path = None
        self.pc_ids = []
        self.pc_paths = []
        if (self.config.network.mode_flag == 'lm') \
                or (self.config.network.mode_flag == 'vae') \
                or (self.config.network.mode_flag == 'all_view') \
                or (self.config.network.mode_flag == 'img_nice'):
            self.split_render_dataset_path = []

        self.set_split_dataset_path()
        self.load_pc_ids()
        if (self.config.network.mode_flag == 'lm') or (self.config.network.mode_flag == 'vae') \
                or (self.config.network.mode_flag == 'img_nice'):
            self.pair_pc_img()
        elif self.config.network.mode_flag == 'all_view':
            self.pair_pc_all_view_img()

    def set_split_dataset_path(self):
        if self.split_dataset_type == 'test':
            self.split_dataset_path = os.path.join(self.config.dataset.dataset_path,
                                                   'valid_models.json')
        else:
            self.split_dataset_path = os.path.join(self.config.dataset.dataset_path,
                                                   str(self.split_dataset_type) + '_models.json')

    def load_pc_ids(self):
        with open(self.split_dataset_path, 'r') as reader:
            jf = json.loads(reader.read())

            if self.config.dataset.train_class is not None:
                self.train_classes = [shapenet_taxonomy.shapenet_category_to_id[class_id]
                                      for class_id in self.config.dataset.train_class]
            else:
                self.train_classes = list(jf.keys())

            if self.config.dataset.test_unseen_flag:
                for train_class in self.train_classes:
                    _ = jf.pop(train_class)
                self.train_classes = list(jf.keys())

            for pc_class in self.train_classes:
                for pc_class_with_id in jf[pc_class]:
                    self.pc_ids.append(pc_class_with_id)

            if self.config.dataset.get_dataset_num(self.split_dataset_type) < len(self.pc_ids):
                random.shuffle(self.pc_ids)
                self.pc_ids = \
                    self.pc_ids[:self.config.dataset.get_dataset_num(self.split_dataset_type)]

            if (self.config.network.mode_flag == 'ae') or (self.config.network.mode_flag == 'nice'):
                for pc_id in self.pc_ids:
                    self.pc_paths.append(os.path.join(self.config.dataset.dataset_path + 'ShapeNet_pointclouds/', pc_id,
                                                      'pointcloud_2048.npy'))

    def pair_pc_img(self):
        num_views = 24
        png_files = [(str(i).zfill(2) + '.png') for i in range(num_views)]
        for id_with_view in itertools.product(self.pc_ids, png_files):
            pc_path = os.path.join(self.config.dataset.dataset_path + 'ShapeNet_pointclouds/',
                                   id_with_view[0], 'pointcloud_2048.npy')
            img_path = os.path.join(self.config.dataset.dataset_path + 'ShapeNetRendering/',
                                    id_with_view[0], 'rendering', id_with_view[1])
            self.split_render_dataset_path.append([pc_path, img_path])

        if self.config.dataset.get_dataset_num(self.split_dataset_type) < len(self.split_render_dataset_path):
            random.shuffle(self.split_render_dataset_path)
            self.split_render_dataset_path = \
                self.split_render_dataset_path[:self.config.dataset.get_dataset_num(self.split_dataset_type)]

    def pair_pc_all_view_img(self):
        num_views = 24
        png_files = [(str(i).zfill(2) + '.png') for i in range(num_views)]
        for pc_id in self.pc_ids:
            pc_path = os.path.join(self.config.dataset.dataset_path + 'ShapeNet_pointclouds/',
                                   pc_id,
                                   'pointcloud_{}.npy'.format(self.config.dataset.resample_amount))
            imgs_in_one_id = []
            for view in png_files:
                img_path = os.path.join(self.config.dataset.dataset_path + 'ShapeNetRendering/',
                                        pc_id, 'rendering', view)
                imgs_in_one_id.append(img_path)
            self.split_render_dataset_path.append([pc_path, imgs_in_one_id])

        if self.config.dataset.get_dataset_num(self.split_dataset_type) < len(self.split_render_dataset_path):
            random.shuffle(self.split_render_dataset_path)
            self.split_render_dataset_path = \
                self.split_render_dataset_path[:self.config.dataset.get_dataset_num(self.split_dataset_type)]
