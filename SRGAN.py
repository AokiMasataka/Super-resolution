import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models

import matplotlib.pyplot as plt


EPOCHS = 32
BATCH_SIZE = 128


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

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.pixelShuffle = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(16, 3, kernel_size=7, padding=3),
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


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.contentLayers = nn.Sequential(*list(vgg.features)[:31]).cuda().eval()
        for param in self.contentLayers.parameters():
            param.requires_grad = False

    def forward(self, fakeFrame, frameY):
        MSELoss = nn.MSELoss()
        content_loss = MSELoss(self.contentLayers(fakeFrame), self.contentLayers(frameY))
        return content_loss


def train(x, y):
    tensor_x, tensor_y = torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
    DS = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(DS, batch_size=BATCH_SIZE, shuffle=True)
    D.train()
    G.train()

    D_optimizer = torch.optim.Adam(D.parameters(), lr=DiscriminatorLR, betas=(0.9, 0.999))
    G_optimizer = torch.optim.Adam(G.parameters(), lr=GeneratorLR, betas=(0.9, 0.999))

    realLabel = torch.ones(BATCH_SIZE, 1).cuda()
    fakeLabel = torch.zeros(BATCH_SIZE, 1).cuda()
    BCE = torch.nn.BCEWithLogitsLoss()
    VggLoss = VGGLoss()


    l = len(x)
    iterate = int(l / BATCH_SIZE)
    for batch_idx, (X, Y) in enumerate(loader):
        if batch_idx == iterate:
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
    genImgs = genImgs / 2 +0.5
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

    fig.savefig("images/gen_%d.png" % epoch)
    plt.close()


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    G = Generator()
    D = Discriminator()

    G = G.cuda()
    D = D.cuda()

    GeneratorLR = 0.00025
    DiscriminatorLR = 0.00001

    X = np.load('data_set')
    Y = np.load('data_set')

    train_x = (X[:-10] / 127.5) - 1
    train_y = (Y[:-10] / 127.5) - 1

    test_x = X[-5:]
    test_y = Y[-5:]
    del X, Y

    for epoch in range(EPOCHS):
        print("epoch : ", epoch)

        train(train_x, train_y)
        print("eval...")
        save_imgs(epoch, test_x, test_y)
