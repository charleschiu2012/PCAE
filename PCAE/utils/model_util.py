import torch
import re
import collections
import torch.distributed
import logging
from glob import glob


def cleanup():
    torch.distributed.destroy_process_group()


class ModelUtil:
    def __init__(self, config):
        self.config = config

    def set_model_parallel_gpu(self, model):
        if self.config.cuda.is_parallel:
            gpu_ids = self.config.cuda.parallel_gpu_ids
            model_with_gpu = None
            if self.config.cuda.dataparallel_mode == 'Dataparallel':
                model_with_gpu = torch.nn.DataParallel(model, device_ids=gpu_ids)
            elif self.config.cuda.dataparallel_mode == 'DistributedDataParallel':
                model_with_gpu = torch.nn.parallel.DistributedDataParallel(model,
                                                                           device_ids=[torch.cuda.current_device()],
                                                                           output_device=self.config.cuda.rank[0])
            return model_with_gpu

    def set_model_device(self, model):
        if self.config.cuda.dataparallel_mode == 'Dataparallel':
            return model.to(self.config.cuda.device)
        elif self.config.cuda.dataparallel_mode == 'DistributedDataParallel':
            return model.to(self.config.cuda.device)

    def get_pretrained_model_path(self, pretrained_epoch):
        if not pretrained_epoch:
            pretrained_model_paths = glob('%s/epoch*' % self.config.network.checkpoint_path)

            model_path = sorted(pretrained_model_paths)[-1] if pretrained_model_paths else None
        else:
            model_path = '%s/epoch%.3d.pth' % (self.config.network.checkpoint_path, int(pretrained_epoch))

        return model_path

    def load_model_pretrain(self, model, pretrained_epoch, is_scratch):
        if pretrained_epoch and is_scratch:
            raise ValueError('Cannot take both argument \'pretrained_model\' and \'scratch\'!')

        model_path = self.get_pretrained_model_path(pretrained_epoch)

        if model_path and (not is_scratch):
            epoch = self.get_epoch_num(model_path)
            if self.config.cuda.dataparallel_mode == 'Dataparallel':
                model.load_state_dict(state_dict=torch.load(f=model_path))
            elif self.config.cuda.dataparallel_mode == 'DistributedDataParallel':
                # Use a barrier() to make sure that process 1 loads the model after process
                # 0 saves it.
                torch.distributed.barrier()
                # self.configure map_location properly
                map_location = {'cuda:%d' % 0: 'cuda:%d' % self.config.cuda.rank}
                model.load_state_dict(state_dict=torch.load(f=model_path, map_location=map_location))
            logging.info('Use pretrained model %s to continue training' % model_path)
            return epoch
        else:
            logging.info('Train from scratch')
            return int(1)

    def test_trained_model(self, model, test_epoch):
        if not test_epoch:
            trained_model_paths = glob('%s/epoch*' % self.config.network.checkpoint_path)
            model_path = sorted(trained_model_paths)[-1] if trained_model_paths else None
        else:
            model_path = '%s/epoch%.3d.pth' % (self.config.network.checkpoint_path, int(test_epoch))

        _ = self.get_epoch_num(model_path)

        if self.config.cuda.dataparallel_mode == 'Dataparallel':
            model.load_state_dict(state_dict=torch.load(f=model_path))
        elif self.config.cuda.dataparallel_mode == 'DistributedDataParallel':
            # Use a barrier() to make sure that process 1 loads the model after process
            # 0 saves it.
            torch.distributed.barrier()
            # self.configure map_location properly
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.config.cuda.rank}
            model.load_state_dict(state_dict=torch.load(f=model_path, map_location=map_location))
        logging.info('Use %s to test' % model_path)

    def load_prior_model(self, model):
        prior_model_path = '../data/LMNet-data/checkpoint/DDP/LMNetAE_half_class/epoch%.3d.pth' % \
                           int(self.config.network.prior_epoch)
        # prior_model_path = '../data/LMNet-data/checkpoint/DDP/LMNetAE/epoch%.3d.pth' % \
        #                    int(self.config.network.prior_epoch)
        if self.config.cuda.dataparallel_mode == 'Dataparallel':
            model.load_state_dict(state_dict=torch.load(f=prior_model_path))
        elif self.config.cuda.dataparallel_mode == 'DistributedDataParallel':
            # Use a barrier() to make sure that process 1 loads the model after process
            # 0 saves it.
            torch.distributed.barrier()
            # configure map_location properly
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.config.cuda.rank}
            model.load_state_dict(state_dict=torch.load(f=prior_model_path, map_location=map_location))
        return model

    @staticmethod
    def get_epoch_num(model_path: str):
        assert isinstance(model_path, str)

        epoch_num_str = re.findall(r'epoch(.+?)\.pth', model_path)
        if epoch_num_str:
            return int(epoch_num_str[0]) + 1
        raise ValueError('Cannot find weights in model path: %s' % model_path)

    @staticmethod
    def get_models_path(checkpoint_path):
        return sorted(glob('%s/epoch*' % checkpoint_path))

    @staticmethod
    def load_partial_pretrained_model(pretrained_model, apply_model, which_part: str):
        apply_part = collections.OrderedDict()
        for k, v in pretrained_model.state_dict().items():
            if k.split('.')[1] == which_part:
                key_name = 'module.' + k.split('.', 2)[2]
                apply_part[key_name] = v

        apply_model_dict = apply_model.state_dict()
        apply_model_dict.update(apply_part)
        apply_model.load_state_dict(apply_model_dict)

        return apply_model

    @staticmethod
    def cleanup():
        torch.distributed.destroy_process_group()
