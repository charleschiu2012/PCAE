from tensorboardX import SummaryWriter
from PCAE.config import config
from PCAE.dataloader import PCDataset
from PCAE.utils.shapenet_taxonomy import shapenet_id_to_category


def add_tsne_data(latent_list, latent_data):
    if config.cuda.device == 'cpu':
        data = latent_data.detach().numpy() if latent_data.requires_grad else latent_data.numpy()
    else:
        data = latent_data.detach().cpu().numpy() if latent_data.requires_grad else latent_data.cpu().numpy()

    latent_list.extend(data)

    return latent_list


def add_tsne_label(label_list, label):
    labels = []
    for i in range(len(label)):
        label_id = label[i].split('/')[0]
        label_category = shapenet_id_to_category[label_id]
        labels.append(label_category)

    label_list.extend(labels)

    return label_list


def visualize_tsne(latent_list, label_list, job_type):
    writer = SummaryWriter(config.tensorboard_dir)
    print('latent_num: ', len(latent_list))
    print('label_num: ', len(label_list))

    if config.network.mode_flag == 'ae':
        writer.add_embedding(latent_list, label_list,
                             tag='{}/{}/{}/{}'.format(config.network.prior_model, config.cuda.dataparallel_mode,
                                                      job_type, PCDataset(job_type).__len__()))
    elif config.network.mode_flag == 'lm':
        writer.add_embedding(latent_list, label_list,
                             tag='{}/{}/{}/{}'.format(config.network.prior_model, config.cuda.dataparallel_mode,
                                                      job_type, PCDataset(job_type).__len__()))

    writer.close()