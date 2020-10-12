import os
import torch
import math
from ...config import config
from ...dataloader import PCDataset


class Network:
    def __init__(self, data_loader, data_type, loss_func=None, optimizer=None, epoch=1, model=None):
        self.model = model
        self._data_loader = data_loader
        self._data_type = data_type
        self._loss_function = loss_func
        self._optimizer = optimizer
        self._epoch = epoch

        self._features = None

    def save_model(self):
        checkpoint_path = config.network.checkpoint_path
        model_name = None
        if config.network.mode_flag == 'ae':
            model_name = str(config.network.prior_model)
        elif config.network.mode_flag == 'lm':
            model_name = str(config.network.img_encoder)
        os.makedirs(checkpoint_path + '/' + model_name, exist_ok=True)
        model_path = '%s/%s/epoch%.3d.pth' % (checkpoint_path,
                                              model_name, int(self._epoch))
        if config.cuda.dataparallel_mode == 'Dataparallel':
            torch.save(self.model.state_dict(), model_path)
        if config.cuda.dataparallel_mode == 'DistributedDataParallel' and config.cuda.rank[0] == 0:
            torch.save(self.model.state_dict(), model_path)

    def get_data(self):
        for i, data in enumerate(self._data_loader):
            device = config.cuda.device
            if config.network.mode_flag == 'ae':
                inputs_pc, targets = data[0].to(device).float().permute(0, 2, 1).contiguous(), \
                                     data[1].to(device).float()
                pc_id = data[2]

                yield inputs_pc, targets, pc_id
            elif config.network.mode_flag == 'lm':
                inputs_img, inputs_pc, targets = data[0].to(device).float().permute(0, 3, 1, 2).contiguous(), \
                                                 data[1].to(device).float().permute(0, 2, 1).contiguous(), \
                                                 data[2].to(device).float()

                img_id, pc_id = data[3], data[4]
                yield inputs_img, inputs_pc, targets, img_id, pc_id

    def steps_in_an_epoch(self):
        if PCDataset(self._data_type).__len__() <= config.network.batch_size:
            return 1
        else:
            return math.ceil(PCDataset(self._data_type).__len__() / config.network.batch_size)

    @property
    def loss_func(self):
        if self._loss_function is None:
            raise ValueError('Loss Function of network not set!')
        return self._loss_function

    @property
    def optimizer(self):
        # learning rate schedule
        # lr = None
        # if self._epoch < math.ceil(config.network.epoch_num/3):
        #     lr = config.network.learning_rate
        # if (self._epoch >= math.ceil(config.network.epoch_num/3)) and \
        #         (self._epoch < math.ceil(2*config.network.epoch_num/3)):
        #     lr = config.network.learning_rate / 10.0
        # if self._epoch >= math.ceil(2*config.network.epoch_num/3):
        #     lr = config.network.learning_rate / 100.0
        lr = config.network.learning_rate

        if self._optimizer is None:
            raise ValueError('Optimizer of network not set!')
        elif self._optimizer == 'Adam':
            self._optimizer = torch.optim.Adam(params=self.model.parameters(),
                                               lr=lr,
                                               betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        elif self._optimizer == 'SGD':
            self._optimizer = torch.optim.SGD(params=self.model.parameters(),
                                              lr=lr,
                                              momentum=config.network.momentum,
                                              weight_decay=0, nesterov=False)

        return self._optimizer

    @property
    def data_loader(self):
        if self._data_loader is None:
            raise ValueError('Dataloader of network not set!')
        return self._data_loader

    @staticmethod
    def tensor_to_numpy(tensor_array):
        assert isinstance(tensor_array, torch.Tensor)

        tensor_array = tensor_array.clone()
        if tensor_array.requires_grad:
            tensor_array = tensor_array.detach()
        if config.cuda.device != 'cpu':
            tensor_array = tensor_array.cpu()

        numpy_array = tensor_array.numpy()
        return numpy_array
