from tensorboardX import SummaryWriter

from PCAE.utils.shapenet_taxonomy import shapenet_id_to_category


class TsneUtil:
    def __init__(self, config):
        self.config = config

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

    def visualize_tsne(self, latent_list, label_list, job_type, _tag):
        writer = SummaryWriter(self.config.tensorboard_dir)
        print('latent_num: ', len(latent_list))
        print('label_num: ', len(label_list))
        assert len(latent_list) == len(label_list)

        writer.add_embedding(latent_list, label_list, tag='{}/{}'.format(_tag, job_type))

        writer.close()
