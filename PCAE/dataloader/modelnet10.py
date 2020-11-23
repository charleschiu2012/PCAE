import glob
import h5py
import numpy as np
import torch
from numpy.random.mtrand import choice
from torch.utils.data import Dataset


class ModelNet10(Dataset):
    def __init__(self, root_dir, subset='train', num_max=2048, num_sample=512):
        self.data = self.load_h5(glob.glob(root_dir + subset + '*.h5'))
        np.random.seed(100)
        self.permutation = choice(num_max, num_sample, replace=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pcd = self.data[idx]
        pcd = torch.from_numpy(pcd).float()
        pcd = pcd[self.permutation]
        return pcd

    @staticmethod
    def load_h5(paths):
        data = []
        for path in paths:
            f = h5py.File(path, 'r')
            data = data + (list(f['data']))
        return data
