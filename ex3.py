import numpy as np
import scipy
import sys
from datetime import datetime

IMAGE_SIZE = 784
FIRST_LAYER_SIZE = 150
LAST_LAYER_SIZE = 10
ETA = 0.005
#EPOCHS = 50
EPOCHS = 50
MAX_COLOR = 255
TRAIN_PERCENT = 0.8

sigmoid = lambda x: 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(np.zeros(np.shape(x)), x)
def relu_derivative(x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return x

def status(msg):
    now = datetime.now()
    print(msg, "\t\t",  now.strftime("%H:%M:%S.%f"))

def softmax(x):
    exps = np.exp(x)
    sum_exps = np.sum(exps)
    # to make sure we don't divide by 0
    if sum_exps == 0:
        sum_exps = 0.001
    return exps / sum_exps

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
                #
                # forward prop
                #
                image = image.reshape(IMAGE_SIZE, 1)
                z1 = np.dot(self.w1, image) + self.b1
                h1 = relu(z1)
                #h1 = sigmoid(z1)
                z2 = np.dot(self.w2, h1) + self.b2
                h2 = softmax(z2)
                predictionsVector = np.zeros(LAST_LAYER_SIZE)
                #  maybe need to remove np.int
                predictionsVector[int(digit)] = 1
                #
                # back prop
                #
                # maybe need to split (l. 90)
                predictionsVector = np.asarray(predictionsVector).reshape(LAST_LAYER_SIZE, 1)

                dl_dz2 = np.subtract(h2, predictionsVector)
                dl_dw2 = np.dot(dl_dz2, np.transpose(h1))
                dl_db2 = dl_dz2
                dl_dh1 = np.dot(np.transpose(self.w2), dl_dz2)
                #dl_dz1 = np.dot(self.w2.T, (dz2)) * sigmoid(z1) * (1 - sigmoid(z1))
                dl_dz1 = np.multiply(dl_dh1, relu_derivative(z1))
                dl_dw1 = np.dot(dl_dz1, np.transpose(image))
                dl_db1 = dl_dz1
                # update params
                self.w1 = np.subtract(self.w1, np.multiply(dl_dw1, ETA))
                self.b1 = np.subtract(self.b1, np.multiply(dl_db1, ETA))
                self.w2 = np.subtract(self.w2, np.multiply(dl_dw2, ETA))
                self.b2 = np.subtract(self.b2, np.multiply(dl_db2, ETA))

    def predict_y_hat(self, image):
        image.shape = (IMAGE_SIZE, 1)
        z1 = np.dot(self.w1, image) + self.b1
        h1 = relu(z1)
        z2 = np.dot(self.w2, h1) + self.b2
        h2 = softmax(z2)
        return np.argmax(h2)


    def test(self):
        status("Test started")
        # array that will store all predictions - this will be the output in the test_y.txt file
        predictions_arr = []
        for image in self.test_x:
            image = np.divide(image, MAX_COLOR)
            predictions_arr.append(int(self.predict_y_hat(image)))
        return predictions_arr


if __name__ == '__main__':
    status("Started")
    nn = NN()
    nn.train()
    predictions = nn.test()
    np.savetxt("test_y", predictions, fmt='%d', delimiter='\n')
    status("Finish")
