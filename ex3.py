import numpy as np
import scipy
import sys

IMAGE_SIZE = 784
FIRST_LAYER_SIZE = 150
LAST_LAYER_SIZE = 10
ETA = 0.005
EPOCHS = 50
MAX_COLOR = 255
TRAIN_PERCENT = 0.8

class NN:
    def __init__(self):
        self.initData()
        self.initParams()


    def initParams(self):
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.w1 = 0.2 * np.random.rand(FIRST_LAYER_SIZE, IMAGE_SIZE) - 0.1
        self.b1 = 0.2 * np.random.rand(FIRST_LAYER_SIZE, 1) - 0.1
        self.w2 = 0.2 * np.random.rand(LAST_LAYER_SIZE, FIRST_LAYER_SIZE) - 0.1
        self.b2 = 0.2 * np.random.rand(LAST_LAYER_SIZE, 1) - 0.1

    def initData(self):
        train_x = np.loadtxt(sys.argv[1])
        train_y = np.loadtxt(sys.argv[2])
        test_x = np.loadtxt(sys.argv[3])

        shufller = np.random.permutation(len(train_x))
        shuffled_train_x = train_x[shufller]
        shuffled_train_y = train_y[shufller]
        self.train_x = shuffled_train_x
        self.train_y = shuffled_train_y

    def train(self):
        for e in range(EPOCHS):
            pass






if __name__ == '__main__':
    nn = NN()
    nn.train()


    print("Check")
