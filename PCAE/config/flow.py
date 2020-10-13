class FlowConfig:
    def __init__(self,
                 ae_dataset_size: dict,
                 lm_dataset_size: dict,
                 train_class,
                 batch_size: int,
                 iter,
                 n_flow: int,
                 n_block: int,
                 lu_flag: bool,
                 affine_flag: bool,
                 n_bits: int,
                 lr: float,
                 temp: float,
                 n_sample: int,
                 ):
        self.batch_size = batch_size
        self.ae_dataset_size = ae_dataset_size
        self.lm_dataset_size = lm_dataset_size
        self.train_class = train_class
        self.iter = int(iter)
        self.n_flow = n_flow  # number of flows in each block
        self.n_block = n_block  # number of blocks
        self.lu_flag = lu_flag  # use plain convolution instead of LU decomposed version
        self.affine_flag = affine_flag  # use affine coupling instead of additive
        self.n_bits = n_bits  # number of bits
        self.n_bins = 2.0 ** self.n_bits  # binary decode from n_bits
        self.lr = lr
        self.temp = temp  # temperature of sampling
        self.n_sample = n_sample  # number of samples
