import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from collections import OrderedDict


class LMImgEncoder(nn.Module):
    def __init__(self, latent_size):
        super(LMImgEncoder, self).__init__()
        self.latent_size = latent_size
        # 128 128
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # 64 64
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # 32 32
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # 16 16
        self.conv10 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv11 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv12 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # 8 8
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv14 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv16 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))

        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.bn4 = torch.nn.BatchNorm2d(64)
        self.bn5 = torch.nn.BatchNorm2d(64)
        self.bn6 = torch.nn.BatchNorm2d(128)
        self.bn7 = torch.nn.BatchNorm2d(128)
        self.bn8 = torch.nn.BatchNorm2d(128)
        self.bn9 = torch.nn.BatchNorm2d(256)
        self.bn10 = torch.nn.BatchNorm2d(256)
        self.bn11 = torch.nn.BatchNorm2d(256)
        self.bn12 = torch.nn.BatchNorm2d(512)
        self.bn13 = torch.nn.BatchNorm2d(512)
        self.bn14 = torch.nn.BatchNorm2d(512)
        self.bn15 = torch.nn.BatchNorm2d(512)
        self.bn16 = torch.nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(4*4*512, self.latent_size)

    def forward(self, x):
        batch_size = x.size()[0]
        # bn=32
        # print("                 b,  ch,   w,  h")
        # print('input', x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        # print('conv1', x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # print('conv2', x.shape)
        # bn=64
        x = F.relu(self.bn3(self.conv3(x)))
        # print('conv3', x.shape)
        x = F.relu(self.bn4(self.conv4(x)))
        # print('conv4', x.shape)
        x = F.relu(self.bn5(self.conv5(x)))
        # print('conv5', x.shape)
        # bn=128
        x = F.relu(self.bn6(self.conv6(x)))
        # print('conv6', x.shape)
        x = F.relu(self.bn7(self.conv7(x)))
        # print('conv7', x.shape)
        x = F.relu(self.bn8(self.conv8(x)))
        # print('conv8', x.shape)
        # bn=256
        x = F.relu(self.bn9(self.conv9(x)))
        # print('conv9', x.shape)
        x = F.relu(self.bn10(self.conv10(x)))
        # print('conv10', x.shape)
        x = F.relu(self.bn11(self.conv11(x)))
        # print('conv11', x.shape)
        # bn=512
        x = F.relu(self.bn12(self.conv12(x)))
        # print('conv12', x.shape)
        x = F.relu(self.bn13(self.conv13(x)))
        # print('conv13', x.shape)
        x = F.relu(self.bn14(self.conv14(x)))
        # print('conv14', x.shape)
        x = F.relu(self.bn15(self.conv15(x)))
        # print('conv15', x.shape)
        x = F.relu(self.bn16(self.conv16(x)))
        # print('conv16', x.shape)

        x = x.reshape(batch_size, -1)
        image_latent = self.fc1(x)
        # print('latent', image_latent.shape)

        return image_latent


class ImgEncoderVAE(nn.Module):
    def __init__(self, latent_size, z_dim):
        super().__init__()
        self.latent_size = latent_size
        self.z_dim = z_dim

    def encoder(self, x):
        encoder = LMImgEncoder(self.latent_size)
        latent = encoder(x)
        fc_mean = nn.Linear(self.latent_size, self.z_dim)
        fc_log_var = nn.Linear(self.latent_size, self.z_dim)

        return fc_mean(latent), fc_log_var(latent)  # mean log_var

    def sampler(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)  # return z sample

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampler(mu, log_var)

        return z, mu, log_var


if __name__ == '__main__':
    x = torch.randn(24, 3, 128, 128)
    model = LMImgEncoder(latent_size=512)
    print(model(x).shape)

    x = torch.randn(24, 3, 128, 128)
    model = ImgEncoderVAE(latent_size=512, z_dim=128)
    print(model(x).shape)



