import open3d as o3d
import numpy as np


def concentroid(o3d_pcd: o3d.geometry.PointCloud):
    return o3d_pcd.get_center()


def max_point(o3d_pcd: o3d.geometry.PointCloud):
    return o3d_pcd.get_max_bound()


def max_length(o3d_pcd: o3d.geometry.PointCloud):
    return np.sum(max_point(o3d_pcd) ** 2, axis=0)


def normalize_pcd(pc):
    if type(pc) == o3d.geometry.PointCloud:
        assert (pc.dimension() == 3)
        np_pcd = open3dpc_to_array(pc)
    else:
        np_pcd = pc

    centroid = np.mean(np_pcd, axis=0)
    np_pcd = np_pcd - centroid
    np_pcd_max_length = np.max(np.sqrt(np.sum(np_pcd ** 2, axis=1)))
    np_pcd = np_pcd / (2*np_pcd_max_length)
    return np_pcd


def get_point_amount(np_pcd: np.ndarray):
    return np_pcd.shape[0]


def resample_pcd(np_pcd: np.ndarray, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(np_pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(np_pcd.shape[0], size=n - np_pcd.shape[0])])
    return np_pcd[idx[:n]]


def open3dpc_to_array(o3d_pcd: o3d.geometry.PointCloud):
    return np.array(o3d_pcd.points)

