import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models

import matplotlib.pyplot as plt


EPOCHS = 8
BATCH_SIZE = 20


class ResidualDenseBlock(nn.Module):
    def __init__(self, deep):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(deep, deep, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(deep, deep, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(deep, deep, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(deep, deep, kernel_size=3, padding=1)

        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.relu3 = nn.PReLU()
        self.relu4 = nn.PReLU()

        self.convBack = nn.Conv2d(deep, deep, kernel_size=3, padding=1)

    def forward(self, x):
        c = self.relu1(self.conv1(x))
        x = x + c
        c = self.relu2(self.conv2(x))
        x = x + c
        c = self.relu3(self.conv3(x))
        x = x + c
        c = self.relu4(self.conv4(x))
        return self.convBack(c)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu = nn.PReLU()

        self.block1 = ResidualDenseBlock(64)
        self.block2 = ResidualDenseBlock(64)
        self.block3 = ResidualDenseBlock(64)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.pixelShuffle = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(8, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1(x)
        skip = self.relu(x)

        x = self.block1(skip) + x
        x = self.block2(x) + x
        x = self.block3(x) + x

        x = self.conv2(x + skip)
        x = self.pixelShuffle(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, size=64):
        super(Discriminator, self).__init__()
        size = int(size / 8) ** 2

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            Flatten(),
            nn.Linear(128 * size, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        return self.net(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


def relative(real, fake):
    Fmean = torch.mean(fake)
    Rmean = torch.mean(real)
    real = torch.sigmoid(real - Fmean)
    fake = torch.sigmoid(fake - Rmean)
    return real, fake


def generator_loss(fakeFrame, frameY, DFake, realLabel):
    vgg = models.vgg16(pretrained=True)
    contentLayers = nn.Sequential(*list(vgg.features)[:31]).cuda().eval()
    for param in contentLayers.parameters():
        param.requires_grad = False

    MSELoss = nn.MSELoss()
    content_loss = MSELoss(contentLayers(fakeFrame), contentLayers(frameY))

    BCELoss = nn.BCELoss()
    adversarial_loss = BCELoss(DFake, realLabel)
    print(adversarial_loss)

    return content_loss + 0.0005 * adversarial_loss # 0.001


def train(x, y):
    tensor_x, tensor_y = torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
    DS = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(DS, batch_size=BATCH_SIZE, shuffle=True)
    D.train()
    G.train()

    realLabel = torch.ones(BATCH_SIZE, 1).cuda()
    fakeLabel = torch.zeros(BATCH_SIZE, 1).cuda()

    for batch_idx, (frameX, frameY) in enumerate(loader):
        frameX = frameX.cuda()
        frameY = frameY.cuda()

        fakeFrame = G(frameX)

        D.zero_grad()
        DReal = D(frameY)
        DFake = D(fakeFrame)
        DReal, DFake = relative(DReal, DFake)

        D_real_loss = d_loss(DReal, realLabel)
        D_fake_loss = d_loss(DFake, fakeLabel)

        D_loss = D_real_loss + D_fake_loss
        D_loss.backward(retain_graph=True)
        D_optimizer.step()

        G.zero_grad()
        G_loss = generator_loss(fakeFrame, frameY, DFake, realLabel)
        print("G_loss :", G_loss, " D_loss :", D_loss)
        G_loss.backward()
        G_optimizer.step()

def save_imgs(epoch, data, datax2):
    r = 5

    G.eval()
    t_data = (data / 127.5) -1
    genImgs = G(torch.tensor(t_data, dtype=torch.float).cuda())
    genImgs = genImgs.cpu().detach().numpy()
    genImgs = (genImgs + 1) * 127.5

    fig, axs = plt.subplots(3, r)
    for i in range(r):
        axs[0, i].imshow(data[i, :, :, :])
        axs[0, i].axis('off')
        axs[1, i].imshow(genImgs[i, :, :, :])
        axs[1, i].axis('off')
        axs[2, i].imshow(datax2[i, :, :, :])
        axs[2, i].axis('off')

    fig.savefig("generat_images/ESRGAN_32To64/gen_%d.png" % epoch)
    plt.close()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    G = Generator()
    D = Discriminator()

    G = G.cuda()
    D = D.cuda()

    GeneratorLR = 0.0004
    DiscriminatorLR = 0.0001

    d_loss = nn.BCELoss()
    print('model deployed')
    X = np.load('data_set/channel_first/size_32.npy')
    Y = np.load('data_set/channel_first/size_64.npy')
    for epoch in range(EPOCHS):
        D_optimizer = torch.optim.Adam(D.parameters(), lr=DiscriminatorLR, betas=(0.9, 0.999))
        G_optimizer = torch.optim.Adam(G.parameters(), lr=GeneratorLR, betas=(0.9, 0.999))

        train((X[10:] / 127.5) - 1, (Y[10:] / 127.5) - 1)

        save_imgs(epoch, X[10:], Y[10:])

        GeneratorLR *= 0.8
        DiscriminatorLR *= 0.8