# PCAE/loss/__init__
import torch
from kaolin.metrics.pointcloud import chamfer_distance
from kaolin.metrics.pointcloud import f_score
from .EMD.emd_module import emdModule
from .pytorch_ssim import SSIM


def chamfer_distance_loss(p1: torch.Tensor, p2: torch.Tensor, w1: float = 1.0, w2: float = 1.0):
    assert (p1.dim() == p2.dim()), 'S1 and S2 must have the same dimesionality'
    assert (p1.dim() == 3), 'the dimensions of the input must be 2 and 1-dimension for batch size.'
    assert (p1.shape[2] == 3), '3rd dimension of input must be 3(xyz)'
    # input_dim = B*N*3

    cd_loss = chamfer_distance(p1=p1, p2=p2, w1=w1, w2=w2)

    return cd_loss.sum()


def emd_loss(predict, target, eps=1e-4, iters=10):
    # predict = predict.transpose(1, 2)
    # target = target.permute(0, 2, 1)
    dist, emd_idx = emdModule()(predict, target, eps, iters)  # eps, iters

    # return torch.sqrt(dist).mean(1).mean()
    return torch.sqrt(dist).mean()


def ssim_loss(target, predict, window_size=11):
    """bigger SSIM loss the better,value max at 1 to be exactly same."""
    _ssim_loss = SSIM(window_size=window_size)

    loss = 1 - _ssim_loss(target, predict)

    return loss


def f_score_loss(gt_points: torch.Tensor, pred_points: torch.Tensor, radius: float = 0.01, eps=False):
    return f_score(gt_points=gt_points, pred_points=pred_points, radius=radius, eps=eps)


def KLDLoss(mu, log_var):
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) * 5
    # print(kld)

    return kld
