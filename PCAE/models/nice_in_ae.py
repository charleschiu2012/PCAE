import torch.nn as nn
import torch.distributions

from .model_3d import LMEncoder, LMDecoder
from .nice import NICE
# from model_3d import LMEncoder, LMDecoder
# from nice import NICE


class NiceAE(nn.Module):
    def __init__(self, prior, coupling, in_out_dim, mid_dim, hidden, mask_config, num_points):
        super(NiceAE, self).__init__()

        self.pc_encoder = LMEncoder(num_points=num_points)
        self.nice = NICE(prior=prior,
                         coupling=coupling,
                         in_out_dim=in_out_dim,
                         mid_dim=mid_dim,
                         hidden=hidden,
                         mask_config=mask_config)
        self.pc_decoder = LMDecoder(num_points=num_points)

    def forward(self, pointclouds):
        latents = self.pc_encoder(pointclouds)

        log_prob_loss = -self.nice(latents)
        z, _ = self.nice.f(latents)
        flow_latents = self.nice.g(z)

        pointclouds = self.pc_decoder(flow_latents)

        return log_prob_loss, pointclouds


if __name__ == '__main__':
    # model hyperparameters
    sample_size = 64
    coupling = 4
    mask_config = 1.

    (full_dim, mid_dim, hidden) = (1 * 512, 128, 5)
    prior = torch.distributions.Normal(
        torch.tensor(0.).cuda(), torch.tensor(1.).cuda())

    inputs = torch.randn(24, 3, 2048).cuda()
    model = NiceAE(prior=prior,
                   coupling=coupling,
                   in_out_dim=full_dim,
                   mid_dim=mid_dim,
                   hidden=hidden,
                   mask_config=mask_config,
                   num_points=2048).cuda()

    model = torch.nn.DataParallel(model, device_ids=[0])
    log_prob_loss, pointclouds = model(inputs)
    print(log_prob_loss)
    print(pointclouds.shape)
