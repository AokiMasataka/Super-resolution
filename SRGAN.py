import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models

import matplotlib.pyplot as plt


EPOCHS = 32
BATCH_SIZE = 128


class ResidualBlock(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock, self).__init__()
        self.Block = nn.Sequential(
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.BatchNorm2d(nf),
            nn.PReLU(),
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.BatchNorm2d(nf),
        )

    def forward(self, x):
        out = self.Block(x)
        return x + out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu = nn.PReLU()

        self.residualLayer = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock()
        )

        self.pixelShuffle = nn.Sequential(
            nn.Conv2d(64, 64*4, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 3, kernel_size=9, padding=4),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1(x)
        skip = self.relu(x)

        x = self.residualLayer(skip)
        x = self.pixelShuffle(x + skip)
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
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.contentLayers = nn.Sequential(*list(vgg.features)[:31]).cuda().eval()
        for param in self.contentLayers.parameters():
            param.requires_grad = False

    def forward(self, fakeFrame, frameY):
        content_loss = torch.nn.functional.mse_loss(self.contentLayers(fakeFrame), self.contentLayers(frameY))
        return content_loss


def train(loader):
    D.train()
    G.train()

    D_optimizer = torch.optim.Adam(D.parameters(), lr=DiscriminatorLR, betas=(0.9, 0.999))
    G_optimizer = torch.optim.Adam(G.parameters(), lr=GeneratorLR, betas=(0.9, 0.999))

    realLabel = torch.ones(BATCH_SIZE, 1).cuda()
    fakeLabel = torch.zeros(BATCH_SIZE, 1).cuda()
    BCE = torch.nn.BCELoss()
    VggLoss = VGGLoss()

    for batch_idx, (X, Y) in enumerate(loader):
        if X.shape[0] < BATCH_SIZE:
            break

        X = X.cuda()
        Y = Y.cuda()

        fakeFrame = G(X)

        D.zero_grad()
        DReal = D(Y)
        DFake = D(fakeFrame)

        D_loss = (BCE(DFake, fakeLabel) + BCE(DReal, realLabel)) / 2
        D_loss.backward(retain_graph=True)
        D_optimizer.step()

        G.zero_grad()
        G_label_loss= BCE(DFake, realLabel)
        G_loss = VggLoss(fakeFrame, Y) + 1e-3 * G_label_loss

        G_loss.backward()
        G_optimizer.step()

        print("G_loss :", G_loss, " D_loss :", D_loss)


def save_imgs(epoch, data, datax2):
    r = 5

    G.eval()
    t_data = (data / 127.5) -1
    genImgs = G(torch.tensor(t_data, dtype=torch.float).cuda())
    genImgs = genImgs.cpu().detach().numpy()
    genImgs = genImgs / 2 + 0.5
    data = np.transpose(data, (0, 2, 3, 1)) / 255
    genImgs = np.transpose(genImgs, (0, 2, 3, 1))
    datax2 = np.transpose(datax2, (0, 2, 3, 1)) / 255

    fig, axs = plt.subplots(3, r)
    for i in range(r):
        axs[0, i].imshow(data[i, :, :, :])
        axs[0, i].axis('off')
        axs[1, i].imshow(genImgs[i, :, :, :])
        axs[1, i].axis('off')
        axs[2, i].imshow(datax2[i, :, :, :])
        axs[2, i].axis('off')

    fig.savefig("generat_images/gen_%d.png" % epoch)
    plt.close()


if __name__ == '__main__':
    G = Generator()
    D = Discriminator()

    G = G.cuda()
    D = D.cuda()

    GeneratorLR = 0.00025
    DiscriminatorLR = 0.00001

    X = np.load('data')
    Y = np.load('data')

    train_x = (X[:-10] / 127.5) - 1
    train_y = (Y[:-10] / 127.5) - 1

    test_x = X[-6:-1]
    test_y = Y[-6:-1]
    del X, Y

    tensor_x, tensor_y = torch.tensor(train_x, dtype=torch.float), torch.tensor(train_y, dtype=torch.float)
    DS = TensorDataset(tensor_x.cuda(), tensor_y.cuda())
    loader = DataLoader(DS, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        train(loader)
        save_imgs(epoch, test_x, test_y)
