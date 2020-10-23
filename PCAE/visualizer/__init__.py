# PCAE/visualizer/__init__
import wandb
import numpy as np

from ..dataloader import PCDataset, FlowDataset


def launch_multiple_runs():
    # Add this line after end of the previous run to finish logging for that run
    wandb.join()


class WandbVisualizer:

    def __init__(self, config, job_type, model):
        self.config = config
        self.project_name = self.config.wandb.project_name
        self.run_name = self.config.wandb.run_name
        self.step_loss = .0
        self.epoch_loss = .0
        self._model = model
        self.job_type = job_type
        if self.config.network.mode_flag == 'ae':
            self.wandb_config = {'dataset': self.config.dataset.dataset_name,
                                 'job_type': self.job_type,
                                 'dataparallel_mode': self.config.cuda.dataparallel_mode,
                                 'split_dataset_size': PCDataset(self.job_type).__len__(),
                                 'machine': self.config.wandb.machine_id,
                                 'model': self.config.network.prior_model,
                                 'loss_fn': self.config.network.loss_func,
                                 'batch_size': self.config.network.batch_size,
                                 'epoch_num': self.config.network.epoch_num,
                                 'learning_rate': self.config.network.learning_rate,
                                 'momentum': self.config.network.momentum}
        elif self.config.network.mode_flag == 'lm':
            self.wandb_config = {'dataset': self.config.dataset.dataset_name,
                                 'job_type': self.job_type,
                                 'split_dataset_size': PCDataset(self.job_type).__len__(),
                                 'machine': self.config.wandb.machine_id,
                                 'model': self.config.network.img_encoder,
                                 'loss_fn': self.config.network.loss_func,
                                 'batch_size': self.config.network.batch_size,
                                 'epoch_num': self.config.network.epoch_num,
                                 'learning_rate': self.config.network.learning_rate,
                                 'momentum': self.config.network.momentum}
        elif self.config.network.mode_flag == 'nice':
            self.wandb_config = {'dataset': self.config.dataset.dataset_name,
                                 'job_type': self.job_type,
                                 'split_dataset_size': FlowDataset(self.job_type).__len__(),
                                 'machine': self.config.wandb.machine_id,
                                 'model': 'NICE',
                                 'loss_fn': 'NICE_loss',
                                 'batch_size': self.config.nice.batch_size,
                                 'epoch_num': self.config.nice.num_iters,
                                 'learning_rate': self.config.network.learning_rate,
                                 'momentum': self.config.network.momentum}

        self.log_init_parameters()
        self.watch_model()

    def log_init_parameters(self):
        wandb.init(project=self.project_name,
                   name=self.run_name,
                   config=self.wandb_config,
                   dir=self.config.wandb.dir_path,
                   job_type=self.job_type,
                   reinit=True,
                   force=False)

    def watch_model(self):
        wandb.watch(self._model)

    @staticmethod
    def log_step_loss(step_idx, step_loss):
        # 'step': If you want to log to a single history step from lots of different places
        # in your code you can pass a step index. Doesn't do with step_idx
        wandb.log({'step_loss': step_loss, 'step': step_idx})

    def log_epoch_loss(self, epoch_idx, train_epoch_loss=None, valid_epoch_loss=None):
        if self.job_type == 'train':
            wandb.log({'train_epoch_loss': train_epoch_loss, 'epoch': epoch_idx})
        elif self.job_type == 'valid':
            wandb.log({'valid_epoch_loss': valid_epoch_loss, 'epoch': epoch_idx})

    @staticmethod
    def log_point_clouds(predict_pc, target_pc):
        assert isinstance(predict_pc, np.ndarray)
        assert isinstance(target_pc, np.ndarray)

        wandb.log({'predict_point_cloud': wandb.Object3D(predict_pc),
                   'target_point_cloud': wandb.Object3D(target_pc)})

    @staticmethod
    def log_single_image(image) -> np.array:
        # save numpy array to PNG image
        wandb.log({'example': wandb.Image(image, caption="write sth {}".format('123'))})

    @staticmethod
    def log_multiple_image(images) -> np.array:
        # save numpy arrays to PNG images
        wandb.log({'example': [wandb.Image(img) for img in images]})

    @staticmethod
    def log_single_video(video):
        wandb.log({"example": wandb.Video(video, caption="write sth {}".format('123'))})
