class NICEConfig:
    def __init__(self,
                 batch_size: int,
                 latent: str,
                 mid_dim: int,
                 num_iters: int,
                 sample_size: int,
                 coupling: int,
                 mask_config: float):

        self.batch_size = batch_size
        self.latent = latent
        self.mid_dim = mid_dim
        self.num_iters = num_iters
        self.sample_size = sample_size
        self.coupling = coupling
        self.mask_config = mask_config
