import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc = nn.Linear(64, 64 * 4 * 4)

        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1)

        self.final_conv = nn.Conv2d(256, 3, kernel_size=5, stride=1, padding=2)

        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.fc(x)

        x = x.view(-1, 64, 4, 4)

        x = self.act(self.deconv1(x))
        x = self.act(self.deconv2(x))
        x = self.act(self.deconv3(x))

        x = torch.tanh(self.final_conv(x))

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc = nn.Linear(64 * 4 * 4, 1)
        self.dropout = nn.Dropout(0.2)

        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))

        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc(x))
        return x


class GeneratorLarge(nn.Module):
    def __init__(self):
        super(GeneratorLarge, self).__init__()

        self.fc = nn.Linear(128, 128 * 8 * 8)

        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 512, kernel_size=4, stride=2, padding=1)

        self.conv = nn.Conv2d(512, 3, kernel_size=5, stride=1, padding=2)

        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.fc(x)

        x = x.view(-1, 128, 8, 8)

        x = self.act(self.deconv1(x))
        x = self.act(self.deconv2(x))
        x = self.act(self.deconv3(x))

        x = torch.tanh(self.conv(x))

        return x


class DiscriminatorLarge(nn.Module):
    def __init__(self):
        super(DiscriminatorLarge, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc = nn.Linear(128 * 8 * 8, 1)
        self.dropout = nn.Dropout(0.2)

        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))

        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc(x))
        return x


class AlternativeGenerator(nn.Module):
    def __init__(self):
        super(AlternativeGenerator, self).__init__()
        self.fc = nn.Linear(128, 128 * 8 * 8)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.final_conv = nn.Conv2d(256, 3, kernel_size=5, stride=1, padding=2)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 8, 8)

        x = self.upsample(x)
        x = self.act(self.bn1(self.conv1(x)))

        x = self.upsample(x)
        x = self.act(self.bn2(self.conv2(x)))

        x = self.upsample(x)
        x = self.act(self.bn3(self.conv3(x)))

        x = torch.tanh(self.final_conv(x))

        return x
