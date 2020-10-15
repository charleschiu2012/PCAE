# PCAE/loss/__init__
import torch
from kaolin.metrics.point import chamfer_distance
from kaolin.metrics.point import directed_distance
from kaolin.metrics.point import iou
from kaolin.metrics.point import f_score
from ....config import config
from .EMD.emd_module import emdModule


class PCLoss:
    def __init__(self):
        self.loss = None

        self.set_loss_func(config.network.loss_func)

    def set_loss_func(self, loss_func_type):
        if loss_func_type == 'cd':
            self.loss = self.chamfer_distance
        elif loss_func_type == 'directed_distance':
            self.loss = self.directed_distance
        elif loss_func_type == 'iou':
            self.loss = self.iou
        elif loss_func_type == 'f_score':
            self.loss = self.f_score
        elif loss_func_type == 'emd':
            self.loss = self.emd_loss
        elif loss_func_type == 'emd+cd':
            self.loss = self.emd_and_cd

    @staticmethod
    def chamfer_distance(s1: torch.Tensor, s2: torch.Tensor, w1: float = 1.0, w2: float = 1.0):
        assert (s1.dim() == s2.dim()), 'S1 and S2 must have the same dimesionality'
        assert (s1.dim() == 3), 'the dimensions of the input must be 2 and 1-dimension for batch size.'
        assert (s1.shape[2] == 3), '3rd dimension of input must be 3(xyz)'
        # input_dim = B*N*3

        loss = torch.tensor([.0], requires_grad=False).to(config.cuda.device)
        for predict_pc, target_pc in zip(s1, s2):
            cd_loss = chamfer_distance(S1=predict_pc, S2=target_pc, w1=w1, w2=w2)
            loss += cd_loss

        loss /= s1.shape[0]  # divide with size of this batch

        return loss

    @staticmethod
    def emd_loss(predict, target, eps=1e-4, iters=10):
        # predict = predict.transpose(1, 2)
        # target = target.permute(0, 2, 1)
        dist, emd_idx = emdModule()(predict, target, eps, iters)  # eps, iters

        # return torch.sqrt(dist).mean(1).mean()
        return torch.sqrt(dist).mean()

    @staticmethod
    def emd_and_cd(predict, target, eps=1e-4, iters=10):

        dist, _ = emdModule()(predict, target, eps, iters)
        emd_loss = torch.sqrt(dist).mean()

        cd_loss = 0
        for predict_pc, target_pc in zip(predict, target):
            cd_loss += chamfer_distance(S1=predict_pc, S2=target_pc, w1=1.0, w2=1.0)

        cd_loss /= predict.shape[0]

        mixture_loss = (0.003*emd_loss + cd_loss)

        return mixture_loss

    @staticmethod
    def directed_distance(s1: torch.Tensor, s2: torch.Tensor, mean: bool = True):
        return directed_distance(S1=s1, S2=s2, mean=mean)

    @staticmethod
    def iou(points1: torch.Tensor, points2: torch.Tensor, thresh=0.5):
        return iou(points1=points1, points2=points2, thresh=thresh)

    @staticmethod
    def f_score(gt_points: torch.Tensor, pred_points: torch.Tensor, radius: float = 0.01, extend=False):
        return f_score(gt_points=gt_points, pred_points=pred_points, radius=radius, extend=extend)


def IEVAELoss(prior_z, predict_z, mu, log_var):
    bce = torch.nn.functional.binary_cross_entropy(prior_z, predict_z, reduction='sum')
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return bce + kld
