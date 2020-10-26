from tensorboardX import SummaryWriter

from PCAE.dataloader import PCDataset, FlowDataset
from PCAE.utils.shapenet_taxonomy import shapenet_id_to_category

config = None #TODO

def add_tsne_data(latent_list, latent_data):
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


'''NICE Tsne
'''


def add_nice_tsne_data(latent_list, latent_data):
    data = latent_data.detach().cpu().numpy() if latent_data.requires_grad else latent_data.cpu().numpy()

    latent_list.extend(data)

    return latent_list


def add_nice_tsne_label(latent_list, label, sample=None):
    if sample == None:
        length = len(latent_list)
        label_list = [label] * length
    else:
        length = len(latent_list) - sample.shape()[0]
        label_list = [label] * length

    return label_list


def visualize_nice_tsne(latent_list, label_list, job_type):
    writer = SummaryWriter(config.tensorboard_dir)
    print('latent_num: ', len(latent_list))
    print('label_num: ', len(label_list))

    if config.network.mode_flag == 'nice':
        writer.add_embedding(latent_list, label_list,
                             tag='{}/{}/{}/{}'.format('NICE', config.cuda.dataparallel_mode,
                                                      job_type, FlowDataset(job_type).__len__()))

    writer.close()
