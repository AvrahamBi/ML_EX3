import numpy as np
import scipy
import sys
from datetime import datetime

#IMAGE_SIZE = 784
#FIRST_LAYER_SIZE = 150
#LAST_LAYER_SIZE = 10
ETA = 0.005
#EPOCHS = 0
EPOCHS = 50
#MAX_COLOR = 255

sigmoid = lambda x: 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(np.zeros(np.shape(x)), x)

# def relu_derivative(x):
#     x[x > 0] = 1
#     x[x <= 0] = 0
#     return x

def relu_derivative(z):
    return np.greater(z, 0).astype(int)

def status(msg):
    now = datetime.now()
    print(msg, "\t\t",  now.strftime("%H:%M:%S.%f"))

# def softmax(x):
#     exps = np.exp(x)
#     sum_exps = np.sum(exps)
#     # to make sure we don't divide by 0
#     if sum_exps == 0:
#         sum_exps = 0.001
#     return exps / sum_exps

def softmax(param):
    param = param - np.max(param)
    expParam = np.exp(param)
    ret = expParam / expParam.sum(keepdims=True)
    return ret

class NN:
    def __init__(self):
        self.initData()
        self.initParams()

    def initParams(self):
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.w1 = 0.2 * np.random.rand(150, 784) - 0.1
        self.b1 = 0.2 * np.random.rand(150, 1) - 0.1
        self.w2 = 0.2 * np.random.rand(10, 150) - 0.1
        self.b2 = 0.2 * np.random.rand(10, 1) - 0.1

    def initData(self):
        # You can remove the shuffle here because train() shuffle the data again
        self.train_x = np.loadtxt(sys.argv[1])
        status("Train_x loaded")
        self.train_y = np.loadtxt(sys.argv[2])
        self.test_x = np.loadtxt(sys.argv[3])
        status("Data loaded!")

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
                image = np.divide(image, 255)
                #
                # forward prop
                #
                image = image.reshape(784, 1)
                z1 = np.dot(self.w1, image) + self.b1
                #h1 = relu(z1)
                h1 = sigmoid(z1)
                z2 = np.dot(self.w2, h1) + self.b2
                h2 = softmax(z2)
                predictionsVector = np.zeros(10)
                predictionsVector[int(digit)] = 1
                # back prop
                predictionsVector = np.asarray(predictionsVector).reshape(10, 1)

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

    def predict(self, image):
        image.shape = (784, 1)
        z1 = np.dot(self.w1, image) + self.b1
        h1 = relu(z1)
        z2 = np.dot(self.w2, h1) + self.b2
        h2 = softmax(z2)
        return np.argmax(h2)


    def test(self):
        status("Test started")
        predictions = []
        for image in self.test_x:
            image = np.divide(image, 255)
            # remove int
            predictions.append(int(self.predict(image)))
        return predictions


if __name__ == '__main__':
    status("Started")
    nn = NN()
    nn.train()
    predictions = nn.test()
    np.savetxt("test_y", predictions, fmt='%d', delimiter='\n')
    status("Finish")
