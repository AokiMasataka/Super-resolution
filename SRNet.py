import keras
from keras.layers import Reshape, add, BatchNormalization, Permute
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt


class SRGAN:
    def __init__(self):
        self.batchSize = 64
        self.r = 2

        self.imageChannels = 3
        self.imageSize = 32
        self.imageSize2x = 64

        self.generatorFilters = 64

        self.generator = self.buildGenerator()
        self.generator.compile(loss='mean_squared_error',
                              optimizer=Adam(lr=8e-4, beta_1=0.5))


    def residualBlock(self, input):
        x = Conv2D(self.generatorFilters, kernel_size=3, strides=1, padding='same')(input)
        x = BatchNormalization()(x)
        x = PReLU()(x)

        x = Conv2D(self.generatorFilters, kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)

        return add([input, x])

    def buildGenerator(self, r=2):
        inputs = keras.Input(shape=(self.imageSize, self.imageSize, self.imageChannels))

        convd = Conv2D(self.generatorFilters, kernel_size=7, strides=1, padding='same')(inputs)
        convd = PReLU()(convd)

        x = self.residualBlock(convd)
        x = self.residualBlock(x)
        x = self.residualBlock(x)

        x = Conv2D(self.generatorFilters, kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization()(x)

        x = add([convd, x])

        x = Conv2D(self.generatorFilters * 4, kernel_size=3, strides=1, padding='same')(x)

        x = Reshape((1, self.imageSize, self.imageSize, r, r, self.generatorFilters))(x)
        x = Permute((1, 2, 4, 3, 5, 6))(x)
        x = Reshape((self.imageSize2x, self.imageSize2x, self.generatorFilters))(x)

        x = PReLU()(x)
        out = Conv2D(self.imageChannels, kernel_size=7, strides=1, padding='same', activation="sigmoid")(x)

        return keras.Model(inputs=inputs, outputs=out)


    def train(self, epochs, batchSize):
        X = np.load('data')
        Y = np.load('data')

        X = X.astype(np.float32) / 255
        X = X.reshape([-1, self.imageSize, self.imageSize, self.imageChannels])

        Y = Y.astype(np.float32) / 255
        Y = Y.reshape([-1, self.imageSize2x, self.imageSize2x, self.imageChannels])

        iterate = int(len(X) / batchSize)

        testPos = -1
        test = 5

        for epoch in range(epochs + 1):
            for i in range(iterate):
                pos = (i * batchSize) + 1
                x1 = X[pos: pos + batchSize]
                x2 = Y[pos: pos + batchSize]

                loss = self.generator.train_on_batch(x1, x2)
                print('epoch :', epoch, ' Loss :', loss)

            self.save_imgs(epoch, X[testPos - test:testPos], Y[testPos - test:testPos])


    def save_imgs(self, epoch, data, datax2):
        r = 5

        genImgs = self.generator.predict(data)

        fig, axs = plt.subplots(3, r)
        for i in range(r):
            axs[0, i].imshow(data[i, :, :, :])
            axs[0, i].axis('off')
            axs[1, i].imshow(genImgs[i, :, :, :])
            axs[1, i].axis('off')
            axs[2, i].imshow(datax2[i, :, :, :])
            axs[2, i].axis('off')

        fig.savefig("SRNet_images/gen_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = SRGAN()
    gan.train(epochs=30, batchSize=64)