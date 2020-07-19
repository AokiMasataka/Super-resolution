import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models

import matplotlib.pyplot as plt


EPOCHS = 32
BATCH_SIZE = 256


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, padding=1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, padding=1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, padding=1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, padding=1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5 * 0.2 + x


class Generator(nn.Module):
    def __init__(self, nf=64):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, nf, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.blockLayer = nn.Sequential(
            ResidualDenseBlock(),
            ResidualDenseBlock(),
            ResidualDenseBlock(),
        )

        self.pixelShuffle = nn.Sequential(
            nn.Conv2d(nf, nf * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(nf, nf, kernel_size=3, padding=1),
            nn.Conv2d(nf, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1(x)
        skip = self.relu(x)

        x = self.blockLayer(skip)
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
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        return self.net(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(models.vgg16(pretrained=True).features[16:23].eval())
        blocks.append(models.vgg16(pretrained=True).features[23:30].eval())
        for block in blocks:
            for param in block:
                param.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks).cuda()
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False).cuda()
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False).cuda()

    def forward(self, fakeFrame, frameY):
        fakeFrame = (fakeFrame - self.mean) / self.std
        frameY = (frameY - self.mean) / self.std
        loss = 0.0
        x = fakeFrame
        y = frameY
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


def relative(real, fake):
    Fmean = torch.mean(fake)
    Rmean = torch.mean(real)
    real = torch.sigmoid(real - Fmean)
    fake = torch.sigmoid(fake - Rmean)
    return real, fake

def train(loader):
    D.train()
    G.train()

    D_optimizer = torch.optim.Adam(D.parameters(), lr=DiscriminatorLR, betas=(0.9, 0.999))
    G_optimizer = torch.optim.Adam(G.parameters(), lr=GeneratorLR, betas=(0.9, 0.999))

    D_scheduler = optim.lr_scheduler.StepLR(D_optimizer, step_size=1, gamma=0.9)
    G_scheduler = optim.lr_scheduler.StepLR(G_optimizer, step_size=1, gamma=1)

    realLabel = torch.ones(BATCH_SIZE, 1).cuda()
    fakeLabel = torch.zeros(BATCH_SIZE, 1).cuda()

    BCE = torch.nn.BCELoss()
    v_loss = VGGPerceptualLoss()

    for batch_idx, (X, Y) in enumerate(loader):
        if X.shape[0] < BATCH_SIZE:
            break

        fakeFrame = G(X)

        D_optimizer.zero_grad()
        DReal = D(Y)
        DFake = D(fakeFrame)

        DReal, DFake = relative(DReal, DFake)
        D_loss = (BCE(DReal, realLabel) + BCE(DFake, fakeLabel)) / 2

        D_loss.backward(retain_graph=True)
        D_optimizer.step()

        G_optimizer.zero_grad()
        G_label_loss = BCE(DFake, realLabel)
        G_loss = v_loss(fakeFrame, Y) + G_label_loss * 0.005

        G_loss.backward()
        G_optimizer.step()

        print("G_loss :", G_loss, " D_loss :", D_loss)

        D_scheduler.step(batch_idx)
        G_scheduler.step(batch_idx)


def save_imgs(epoch, data, datax2):
    r = 5

    G.eval()
    t_data = (data / 127.5) - 1
    genImgs = G(torch.tensor(t_data, dtype=torch.float).cuda())
    genImgs = genImgs.cpu().detach().numpy()
    genImgs = (genImgs / 2) + 0.5
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

    fig.savefig("images/%d.png" % epoch)
    plt.close()


if __name__ == '__main__':
    G = Generator()
    D = Discriminator()

    G = G.cuda()
    D = D.cuda()
    GeneratorLR = 0.0002
    DiscriminatorLR = 0.00001

    X = np.load('data_set')
    Y = np.load('data_set')
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

