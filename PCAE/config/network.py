class NetworkConfig:
    def __init__(self,
                 mode_flag: str,
                 img_encoder: str,
                 prior_model: str,
                 checkpoint_path: str,
                 prior_epoch: str,
                 loss_scale_factor: int,
                 batch_size: int = 24,
                 latent_size: int = 512,
                 z_dim: int = 128,
                 epoch_num: int = 30,
                 learning_rate: float = 1e-3):

        self.mode_flag = mode_flag
        self.img_encoder = img_encoder
        self.prior_model = prior_model
        self.checkpoint_path = checkpoint_path
        self.prior_epoch = prior_epoch
        self.loss_scale_factor  = loss_scale_factor
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.z_dim = z_dim
        self.epoch_num = epoch_num
        self.learning_rate = learning_rate
