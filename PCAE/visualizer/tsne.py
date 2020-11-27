from tensorboardX import SummaryWriter

from PCAE.utils.shapenet_taxonomy import shapenet_id_to_category


class TsneUtil:
    def __init__(self, config):
        self.config = config
        self.latent_list = []
        self.label_list = []

    def add_tsne_data(self, latent_data):
        data = latent_data.detach().cpu().numpy() if latent_data.requires_grad else latent_data.cpu().numpy()

        self.latent_list.extend(data)

    def add_tsne_label(self, label):
        self.label_list.extend(label)

    def add_tsne_pc_class_label(self, label):
        labels = []
        for i in range(len(label)):
            label_id = label[i].split('/')[0]
            label_category = shapenet_id_to_category[label_id]
            labels.append(label_category)

        self.label_list.extend(labels)

    def visualize_tsne(self, job_type: str, _tag: str):
        writer = SummaryWriter(self.config.tensorboard_dir)
        print('latent_num: ', len(self.latent_list))
        print('label_num: ', len(self.label_list))
        assert len(self.latent_list) == len(self.label_list)

        writer.add_embedding(self.latent_list, self.label_list, tag='{}/{}'.format(_tag, job_type))

        writer.close()
