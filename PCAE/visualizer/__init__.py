# PCAE/visualizer/__init__
import wandb
import numpy as np
import multiprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.express as px

from PCAE.utils.shapenet_taxonomy import shapenet_id_to_category


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
        self.wandb_config = {'dataset': self.config.dataset.dataset_name,
                             'job_type': self.job_type,
                             'dataparallel_mode': self.config.cuda.dataparallel_mode,
                             'split_dataset_size': self.config.dataset.dataset_size[self.job_type],
                             'machine': self.config.wandb.machine_id,
                             'batch_size': self.config.network.batch_size,
                             'epoch_num': self.config.network.epoch_num,
                             'learning_rate': self.config.network.learning_rate}

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

    def log_epoch_loss(self, epoch_idx, loss_type: str, train_epoch_loss=None, valid_epoch_loss=None):
        if self.job_type == 'train':
            wandb.log({'train_{}_epoch_loss'.format(loss_type): train_epoch_loss, 'epoch': epoch_idx})
        elif self.job_type == 'valid':
            wandb.log({'valid_{}_epoch_loss'.format(loss_type): valid_epoch_loss, 'epoch': epoch_idx})

    @staticmethod
    def add_tsne_data(latent_list, latent_data):
        data = latent_data.detach().cpu().numpy() if latent_data.requires_grad else latent_data.cpu().numpy()

        latent_list.extend(data)

        return latent_list

    @staticmethod
    def add_tsne_label(label_list, label):
        labels = []
        for i in range(len(label)):
            label_id = label[i].split('/')[0]
            label_category = shapenet_id_to_category[label_id]
            labels.append(label_category)

        label_list.extend(labels)

        return label_list

    @staticmethod
    def compute_tsne(latent_list, label_list):
        print('latent_num: ', len(latent_list))
        print('label_num: ', len(label_list))
        latent_array = np.array(latent_list)
        label_array = np.array(label_list)
        # num_samples = 10000
        # latent_array = latent_array[:num_samples]
        # label_array = label_array[:num_samples]

        num_components = 100
        pca = PCA(n_components=num_components)
        reduced = pca.fit_transform(latent_array)

        tsne = TSNE(n_components=3, perplexity=25.0, n_jobs=multiprocessing.cpu_count() * 5)
        tsne_result = tsne.fit_transform(reduced)
        tsne_result_scaled = StandardScaler().fit_transform(tsne_result)
        return tsne_result_scaled, label_array

    @staticmethod
    def draw_tsne(tsne_result_scaled, label_array, tsne_name):
        wandb.init(project="TSNE")

        tsne_data = {"X": tsne_result_scaled[:, 0],
                     "Y": tsne_result_scaled[:, 1],
                     "Z": tsne_result_scaled[:, 2],
                     "Label": label_array}

        tsne_df = pd.DataFrame(tsne_data)
        fig = px.scatter_3d(tsne_df, x=tsne_df.X, y=tsne_df.Y, z=tsne_df.Z, color=tsne_df.Label)
        wandb.log({"{}_TSNE".format(tsne_name): fig})

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
