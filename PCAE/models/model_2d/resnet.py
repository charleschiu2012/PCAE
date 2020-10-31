import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50


class PointDecoder(nn.Module):
    def __init__(self, num_points=2048, k=2):
        super(PointDecoder, self).__init__()
        self.num_points = num_points
        self.fc1 = nn.Linear(100, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, self.num_points * 3)
        self.th = nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.th(self.fc5(x))
        x = x.view(batchsize, 3, self.num_points)
        return x


class ResNet18PC(nn.Module):
    def __init__(self, num_points=2048):
        super(ResNet18PC, self).__init__()
        self._features = None
        self._model = resnet18(pretrained=True)
        self.extend_encoder = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
        )

        self.decoder = PointDecoder(num_points)

    def forward(self, x):
        output = self._model.conv1(x)
        output = self._model.bn1(output)
        output = self._model.relu(output)
        output = self._model.maxpool(output)

        output = self._model.layer1(output)
        output = self._model.layer2(output)
        output = self._model.layer3(output)
        output = self._model.layer4(output)
        output = self._model.avgpool(output)
        latent_feat = self.extend_encoder(output)

        output = self.decoder(latent_feat)

        return latent_feat, output


# class ResNet34(nn.Module):
#     def __init__(self):
#         super(ResNet34, self).__init__()
#         self._features = None
#         self._model = resnet34(pretrained=True)
#         self._model.regression = self._make_regression()
#
#     def forward(self, x):
#         output = self._model.conv1(x)
#         output = self._model.bn1(output)
#         output = self._model.relu(output)
#         output = self._model.maxpool(output)
#
#         output = self._model.layer1(output)
#         output = self._model.layer2(output)
#         output = self._model.layer3(output)
#         output = self._model.layer4(output)
#         output = self._model.avgpool(output)
#
#         output = output.view(output.size(0), -1)
#         self._features = output.clone()
#
#         output = self._model.regression(output)
#
#         return output
#
#     @staticmethod
#     def _make_regression():
#         return nn.Sequential(
#             nn.Linear(512, 1024),
#             nn.Dropout(p=0.5),
#             nn.Linear(1024, 512),
#             nn.Dropout(p=0.5),
#             nn.Linear(512, 64),
#             nn.Dropout(p=0.5),
#             nn.Linear(64, len(config.dataset.labels)),
#             NormalizeLayer()
#         )
#
#     @property
#     def features(self) -> np.ndarray:
#         if self._features is None:
#             raise ValueError('Features of VGG19 model is empty!')
#
#         features = self._features
#         if features.requires_frad:
#             features = features.detach()
#
#         if config.cuda.device != 'cpu':
#             features = features.cpu()
#
#         return features.numpy()
#
#
# class ResNet50(nn.Module):
#     def __init__(self):
#         super(ResNet50, self).__init__()
#         self._features = None
#         self._model = resnet50(pretrained=True)
#         self._model.regression = self._make_regression()
#
#     def forward(self, x):
#         output = self._model.conv1(x)
#         output = self._model.bn1(output)
#         output = self._model.relu(output)
#         output = self._model.maxpool(output)
#
#         output = self._model.layer1(output)
#         output = self._model.layer2(output)
#         output = self._model.layer3(output)
#         output = self._model.layer4(output)
#         output = self._model.avgpool(output)
#
#         output = output.view(output.size(0), -1)
#         self._features = output.clone()
#
#         output = self._model.regression(output)
#
#         return output
#
#     @staticmethod
#     def _make_regression():
#         return nn.Sequential(
#             nn.Linear(2048, 1024),
#             nn.Dropout(p=0.5),
#             nn.Linear(1024, 512),
#             nn.Dropout(p=0.5),
#             nn.Linear(512, 64),
#             nn.Dropout(p=0.5),
#             nn.Linear(64, len(config.dataset.labels)),
#             NormalizeLayer()
#         )
#
#     @property
#     def features(self) -> np.ndarray:
#         if self._features is None:
#             raise ValueError('Features of VGG19 model is empty!')
#
#         features = self._features
#         if features.requires_frad:
#             features = features.detach()
#
#         if config.cuda.device != 'cpu':
#             features = features.cpu()
#
#         return features.numpy()
