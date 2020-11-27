import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from collections import OrderedDict
from .LMImgEncoder import LMImgEncoder


class ImgDecoder(nn.Module):
    def __init__(self, latent_size, image_shape):
        super(ImgDecoder, self).__init__()
        self.latent_size = latent_size
        self.image_shape = image_shape

        self.fc1 = nn.Linear(self.latent_size, 4 * 4 * 512)
        self.conv1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=5, stride=2,  padding=1)

        # 8 8
        self.conv2 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        # 16 16
        self.conv5 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        # 32 32
        self.conv8 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv9 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # 64 64
        self.conv11 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv12 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 128 128
        self.conv14 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv15 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv16 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.bn1 = torch.nn.BatchNorm2d(512)
        self.bn2 = torch.nn.BatchNorm2d(512)
        self.bn3 = torch.nn.BatchNorm2d(512)
        self.bn4 = torch.nn.BatchNorm2d(512)
        self.bn5 = torch.nn.BatchNorm2d(256)
        self.bn6 = torch.nn.BatchNorm2d(256)
        self.bn7 = torch.nn.BatchNorm2d(256)
        self.bn8 = torch.nn.BatchNorm2d(128)
        self.bn9 = torch.nn.BatchNorm2d(128)
        self.bn10 = torch.nn.BatchNorm2d(128)
        self.bn11 = torch.nn.BatchNorm2d(64)
        self.bn12 = torch.nn.BatchNorm2d(64)
        self.bn13 = torch.nn.BatchNorm2d(64)
        self.bn14 = torch.nn.BatchNorm2d(32)
        self.bn15 = torch.nn.BatchNorm2d(32)
        self.bn16 = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.reshape(batch_size, -1)
        x = self.fc1(x)
        x = x.reshape(batch_size, 512, 4, 4)
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
        image = F.relu(self.bn16(self.conv16(x)))
        # print('conv16', image.shape)

        return image[:, :, :self.image_shape, :self.image_shape]


class ImgAE(nn.Module):
    def __init__(self, latent_size, image_shape):
        super(ImgAE, self).__init__()
        self.latent_size = latent_size
        self.image_shape = image_shape
        self.encoder = LMImgEncoder(latent_size=self.latent_size)
        self.decoder = ImgDecoder(latent_size=self.latent_size, image_shape=self.image_shape)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


if __name__ == '__main__':
    decoder = ImgDecoder(latent_size=512, image_shape=128)
    x = torch.randn(24, 512)
    print(decoder(x).max())
    print(decoder(x).shape)
    t = torch.randn(24, 3, 128, 128)
    loss = nn.MSELoss()(decoder(x), t)
    loss.backward()