class NICEConfig:
    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 num_iters: int,
                 sample_size: int,
                 latent: str = 'normal',
                 mid_dim: int = 128,
                 coupling: int = 4,
                 mask_config: float = 1.):

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.latent = latent
        self.mid_dim = mid_dim
        self.num_iters = num_iters
        self.sample_size = sample_size
        self.coupling = coupling
        self.mask_config = mask_config
