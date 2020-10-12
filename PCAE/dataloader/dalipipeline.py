import types
import collections
import numpy as np
import copy
from random import shuffle
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from ..config import config
from .dataset import DatasetLoader
from ..utils import pointcloud_util


def get_pc(pc):
    normalized_pc = pointcloud_util.normalize_pcd(pc)
    # resampled_pc = pointcloud_util.resample_pcd(normalized_pc, config.dataset.resample_amount)

    return normalized_pc


class ExternalInputIterator(object):
    def __init__(self, split_dataset_type: str):
        self.dataset_loader = DatasetLoader(split_dataset_type)
        self.split_render_dataset_path = self.dataset_loader.split_render_dataset_path

        shuffle(self.split_render_dataset_path)

    def __iter__(self):
        self.i = 0
        self.n = len(self.split_render_dataset_path)
        return self

    def __next__(self):
        batch_pc_raws = []
        batch_img_raws = []
        if config.network.mode_flag == "ae":
            for _ in range(config.network.batch_size):
                pc_path = self.dataset_loader.split_render_dataset_path[self.i][0]
                pc_f = open(pc_path, 'rb')
                batch_pc_raws.append(np.frombuffer(pc_f.read(), dtype=np.uint8))
                self.i = (self.i + 1) % self.n
            return batch_pc_raws
        elif config.network.mode_flag == "lm":
            for _ in range(config.network.batch_size):
                pc_path = self.dataset_loader.split_render_dataset_path[self.i][0]
                pc_f = open(pc_path, 'rb')
                batch_pc_raws.append(np.frombuffer(pc_f.read(), dtype=np.uint8))
                img_path = self.dataset_loader.split_render_dataset_path[self.i][1]
                img_f = open(img_path, 'rb')
                batch_img_raws.append(np.frombuffer(img_f.read(), dtype=np.uint8))
                self.i = (self.i + 1) % self.n
            return batch_pc_raws, batch_img_raws


class DaliPipeline(Pipeline):
    def __init__(self, batch_size, eii, num_threads, device_id):
        super(DaliPipeline, self).__init__(batch_size, num_threads, device_id, exec_async=False,
                                           exec_pipelined=False, seed=12)

        self.source = ops.ExternalSource(source=eii, num_outputs=2)
        self.load_pc = ops.PythonFunction(function=get_pc, num_outputs=1)
        self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB)
        self.numpy_reader = ops.NumpyReader(device='cpu')
        self.resize = ops.Resize(device='gpu', resize_x=128, resize_y=128)

    def define_graph(self):
        pc_raws, img_raws = self.source()
        pcs = self.numpy_reader(pc_raws)
        normalized_pcs = self.load_pc(pcs)
        targest_pcs = copy.deepcopy(normalized_pcs)
        imgs = self.decode(img_raws)
        imgs = self.resize(imgs)
        return imgs, normalized_pcs, targest_pcs


# if __name__ == '__main__':
#     batch_size = 512
#     eii = ExternalInputIterator('test')
#     pipe = DaliPipeline(batch_size=batch_size, eii=eii, num_threads=40, device_id=0)
#     pipe.build()
#     imgs, normalized_pcs, targest_pcs = pipe.run()
#
#     print(imgs.shape)
#     print(normalized_pcs.shape)
#     print(targest_pcs.shape())