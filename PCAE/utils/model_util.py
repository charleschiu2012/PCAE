import torch
import re
import collections
import torch.distributed
import logging
from glob import glob
from ..config import config


def cleanup():
    torch.distributed.destroy_process_group()


def set_model_parallel_gpu(model):
    if config.cuda.is_parallel:
        gpu_ids = config.cuda.parallel_gpu_ids
        model_with_gpu = None
        if config.cuda.dataparallel_mode == 'Dataparallel':
            model_with_gpu = torch.nn.DataParallel(model, device_ids=gpu_ids)
        elif config.cuda.dataparallel_mode == 'DistributedDataParallel':
            # for param in model.features.parameters():
            #     param.requires_grad = True
            model_with_gpu = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()],
                                                                       output_device=config.cuda.rank[0])
        return model_with_gpu


def set_model_device(model):
    if config.cuda.dataparallel_mode == 'Dataparallel':
        return model.to(config.cuda.device)
    elif config.cuda.dataparallel_mode == 'DistributedDataParallel':
        return model.to(config.cuda.device)


def get_pretrained_model_path(pretrained_epoch):
    model_path = None
    if not pretrained_epoch:
        pretrained_model_paths = None
        if config.network.mode_flag == 'ae':
            pretrained_model_paths = glob('%s/%s/epoch*' % (config.network.checkpoint_path,
                                                            config.network.prior_model))
        elif config.network.mode_flag == 'lm':
            pretrained_model_paths = glob('%s/%s/epoch*' % (config.network.checkpoint_path,
                                                            config.network.img_encoder))
        model_path = sorted(pretrained_model_paths)[-1] if pretrained_model_paths else None
    else:
        if config.network.mode_flag == 'ae':
            model_path = '%s/%s/epoch%.3d.pth' % (config.network.checkpoint_path,
                                                  config.network.prior_model,
                                                  int(pretrained_epoch))
        elif config.network.mode_flag == 'lm':
            model_path = '%s/%s/epoch%.3d.pth' % (config.network.checkpoint_path,
                                                  config.network.img_encoder,
                                                  int(pretrained_epoch))

    return model_path


def load_model_pretrain(model, pretrained_epoch, is_scratch):
    if pretrained_epoch and is_scratch:
        raise ValueError('Cannot take both argument \'pretrained_model\' and \'scratch\'!')

    model_path = get_pretrained_model_path(pretrained_epoch)

    if model_path and (not is_scratch):
        epoch = get_epoch_num(model_path)
        if config.cuda.dataparallel_mode == 'Dataparallel':
            model.load_state_dict(state_dict=torch.load(f=model_path))
        elif config.cuda.dataparallel_mode == 'DistributedDataParallel':
            # Use a barrier() to make sure that process 1 loads the model after process
            # 0 saves it.
            torch.distributed.barrier()
            # configure map_location properly
            map_location = {'cuda:%d' % 0: 'cuda:%d' % config.cuda.rank}
            model.load_state_dict(state_dict=torch.load(f=model_path, map_location=map_location))
        logging.info('Use pretrained model %s to continue training' % model_path)
        return epoch
    else:
        logging.info('Train from scratch')
        return int(1)


def test_trained_model(model, test_epoch):
    model_path = None
    if not test_epoch:
        trained_model_paths = None
        if config.network.mode_flag == 'ae':
            trained_model_paths = glob('%s/%s/epoch*' % (config.network.checkpoint_path, config.network.prior_model))
        elif config.network.mode_flag == 'lm':
            trained_model_paths = glob('%s/%s/epoch*' % (config.network.checkpoint_path, config.network.img_encoder))
        model_path = sorted(trained_model_paths)[-1] if trained_model_paths else None
    else:
        if config.network.mode_flag == 'ae':
            model_path = '%s/%s/epoch%.3d.pth' % (config.network.checkpoint_path,
                                                  config.network.prior_model, int(test_epoch))
        elif config.network.mode_flag == 'lm':
            model_path = '%s/%s/epoch%.3d.pth' % (config.network.checkpoint_path,
                                                  config.network.img_encoder, int(test_epoch))

    _ = get_epoch_num(model_path)

    if config.cuda.dataparallel_mode == 'Dataparallel':
        model.load_state_dict(state_dict=torch.load(f=model_path))
    elif config.cuda.dataparallel_mode == 'DistributedDataParallel':
        # Use a barrier() to make sure that process 1 loads the model after process
        # 0 saves it.
        torch.distributed.barrier()
        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % config.cuda.rank}
        model.load_state_dict(state_dict=torch.load(f=model_path, map_location=map_location))
    logging.info('Use %s to test' % model_path)


def get_epoch_num(model_path: str):
    assert isinstance(model_path, str)

    epoch_num_str = re.findall(r'epoch(.+?)\.pth', model_path)
    if epoch_num_str:
        return int(epoch_num_str[0]) + 1
    raise ValueError('Cannot find weights in model path: %s' % model_path)


def get_models_path(checkpoint_path):
    if config.network.mode_flag == 'ae':
        return sorted(glob('%s/%s/epoch*' % (checkpoint_path, config.network.prior_model)))
    elif config.network.mode_flag == 'lm':
        return sorted(glob('%s/%s/epoch*' % (checkpoint_path, config.network.img_encoder)))


def load_partial_pretrained_model(pretrained_model, epoch_num, apply_model, which_part):
    # checkpoint_path = '%s/%s/epoch%.3d.pth' % (config.network.checkpoint_path,
    #                                            config.network.prior_model, int(epoch_num))
    #
    # if config.cuda.dataparallel_mode == 'Dataparallel':
    #     pretrained_model.load_state_dict(state_dict=torch.load(f=checkpoint_path))
    # elif config.cuda.dataparallel_mode == 'DistributedDataParallel':
    #     # Use a barrier() to make sure that process 1 loads the model after process
    #     # 0 saves it.
    #     torch.distributed.barrier()
    #     # configure map_location properly
    #     map_location = {'cuda:%d' % 0: 'cuda:%d' % config.cuda.rank}
    #     pretrained_model.load_state_dict(state_dict=torch.load(f=checkpoint_path, map_location=map_location))

    apply_part = collections.OrderedDict()
    for k, v in pretrained_model.state_dict().items():
        if k.split('.')[1] == which_part:
            key_name = 'module.' + k.split('.', 2)[2]
            apply_part[key_name] = v

    apply_model_dict = apply_model.state_dict()
    apply_model_dict.update(apply_part)
    apply_model.load_state_dict(apply_model_dict)

    return apply_model
