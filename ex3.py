import numpy as np
import scipy
import sys
from datetime import datetime

IMAGE_SIZE = 784
FIRST_LAYER_SIZE = 150
LAST_LAYER_SIZE = 10
ETA = 0.005
#EPOCHS = 50
EPOCHS = 3
MAX_COLOR = 255
TRAIN_PERCENT = 0.8


def status(msg):
    now = datetime.now()
    print(msg, "\t\t",  now.strftime("%H:%M:%S.%f"))

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
        # You can remove the shuffle here because train() shuffle the data again
        self.train_x = np.loadtxt(sys.argv[1])
        status("Train_x loaded")
        self.train_y = np.loadtxt(sys.argv[2])
        self.test_x = np.loadtxt(sys.argv[3])
        status("Data loaded!")
        # shufller = np.random.permutation(len(train_x))
        # shuffled_train_x = train_x[shufller]
        # shuffled_train_y = train_y[shufller]
        # self.train_x = shuffled_train_x
        # self.train_y = shuffled_train_y


    def relu(self):
        pass

    def softmax(self):
        pass

    def predict_y_hat(self, image):
        image = shape(IMAGE_SIZE, 1)
        z1 = np.dot(w1, image) + b1
        h1 = self.relu(z1)
        z2 = np.dot(w2, h1) + b2
        h2 = self.softmax(z2)
        return np.argmax(h2)



    # Back propagation, Forward propagation
    def train(self):
        status("Training began")
        for e in range(EPOCHS):
            status("Epoch No. " + str(e))
            shufller = np.random.permutation(len(self.train_x))
            train_x = self.train_x[shufller]
            train_y = self.train_y[shufller]
            loss = []
            for image, digit in zip(train_x, train_y):
                image = np.divide(image, MAX_COLOR)
                # forward prop
                image = image.reshape(IMAGE_SIZE, 1)
                z1 = np.dot(self.w1, image) + self.b1
                h1 = relu(z1)
                z2 = np.dot(self.w2, h1) + self.b2
                h2 = softmax(z2)
                predictionsVector = np.zeros(LAST_LAYER_SIZE)
                #  maybe need to remove np.int
                predictionsVector[np.int(digit)] = 1
                # back prop

                # update params









if __name__ == '__main__':
    status("Started")
    nn = NN()
    nn.train()


    status("Finish")
