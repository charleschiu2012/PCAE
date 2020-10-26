import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


class LMEncoder(nn.Module):
    def __init__(self, num_points):
        super(LMEncoder, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = torch.nn.Conv1d(128, 128, kernel_size=1)
        self.conv4 = torch.nn.Conv1d(128, 256, kernel_size=1)
        self.conv5 = torch.nn.Conv1d(256, 512, kernel_size=1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.bn4 = torch.nn.BatchNorm1d(256)
        self.bn5 = torch.nn.BatchNorm1d(512)

        self.mp1 = torch.nn.MaxPool1d(num_points)

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.mp1(x)

        x, _ = torch.max(x, 2)
        x = x.view(batch_size, 512)

        return x


class LMDecoder(nn.Module):
    def __init__(self, num_points):
        super(LMDecoder, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, self.num_points * 3)

        self.bn1 = torch.nn.BatchNorm1d(256)
        self.bn2 = torch.nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        x = x.view(batch_size, self.num_points, 3)
        return x


class LMNetAE(nn.Module):
    def __init__(self, num_points):
        super(LMNetAE, self).__init__()
        self.num_points = num_points
        self.encoder = LMEncoder(num_points)

        self.decoder = LMDecoder(num_points)

    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)

        return latent, x


if __name__ == '__main__':
    x = torch.randn(32, 3, 1024)
    pretrained_model = LMNetAE(num_points=1024)
    latent, result = pretrained_model(x)
    print(latent.shape)
    print(result.shape)
    # print(pretrained_model.state_dict())
    import collections
    decoder_part = collections.OrderedDict()
    print(pretrained_model.state_dict().keys())
    for k, v in pretrained_model.state_dict().items():
        if k.split('.')[0] == 'decoder':
            print(len('decoder'))
            print()
            # print(k)
            decoder_part[k[8:]] = v
    # decoder_keys = [k for k, v in pretrained_model.state_dict() if k.split('.')[0] == 'decoder']
    print(decoder_part.keys())
    decoder = LMDecoder(1024)
    print(decoder.state_dict().keys())
    if list(decoder.state_dict().keys()) == list(decoder_part.keys()):
        print('they have same keys')

    pretrained_model_dict = pretrained_model.state_dict()
    decoder_dict = decoder.state_dict()

    # pretrained_model_dict = {k: v for k, v in pretrained_model_dict.items() if k in decoder_part}
    decoder_dict.update(decoder_part)
    decoder.load_state_dict(decoder_dict)
    for pk, pv in pretrained_model.state_dict().items():
        if pk.split('.')[0] == 'decoder':
            if (pv != decoder.state_dict()[pk[8:]]).all():
                print(pk)

                print('{} and {} are not the same'.format(pk, pk[8:]))
