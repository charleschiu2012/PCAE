import torch


class CudaConfig:
    def __init__(self,
                 device: str = 'cuda',
                 is_parallel: bool = False,
                 parallel_gpu_ids: list = None,
                 dataparallel_mode: str = 'Dataparallel'):

        self.device = device
        self.is_parallel = is_parallel
        self.parallel_gpu_ids = parallel_gpu_ids
        self.dataparallel_mode = dataparallel_mode
        self.rank = None
        self.world_size = None

        self.check_parameters()
        self.setup()

    def setup(self):
        if self.dataparallel_mode == 'DistributedDataParallel':
            torch.distributed.init_process_group(backend='nccl', init_method='env://')
            self.rank = int(torch.distributed.get_rank()),
            self.world_size = int(torch.distributed.get_world_size()),

    def check_parameters(self):
        assert self.device == 'cuda' or 'cpu'
        assert isinstance(self.is_parallel, bool)
