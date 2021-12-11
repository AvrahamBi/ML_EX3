import numpy as np
import sys

LR = 0.005
EPOCHS = 55
#EPOCHS = 1

def relu(x):
    return np.maximum(np.zeros(np.shape(x)), x)

def relu_derivative(z):
    return np.greater(z, 0).astype(int)

def status(msg):
    #pass
    #now = datetime.now()
    print(msg)

def softmax(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / ex.sum(keepdims=True)

class NN:
    def __init__(self):
        self.initData()
        self.initParams()

    def initParams(self):
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.w1 = np.random.rand(150, 784)
        self.b1 = np.random.rand(150, 1)
        self.w2 = np.random.rand(10, 150)
        self.b2 = np.random.rand(10, 1)
        # normalize
        self.w1 = 0.2 * self.w1 - 0.1
        self.b1 = 0.2 * self.b1 - 0.1
        self.w2 = 0.2 * self.w2 - 0.1
        self.b2 = 0.2 * self.b2 - 0.1

    def initData(self):
        # You can remove the shuffle here because train() shuffle the data again
        self.train_x = np.loadtxt(sys.argv[1])
        #status("Train_x loaded")
        self.train_y = np.loadtxt(sys.argv[2])
        self.test_x = np.loadtxt(sys.argv[3])
        #status("Data loaded!")

    def train(self):
        #status("Training began")
        for e in range(EPOCHS):
            #status("Epoch No. " + str(e))
            shufller = np.random.permutation(len(self.train_x))
            train_x = self.train_x[shufller]
            train_y = self.train_y[shufller]
            loss = []
            for image, digit in zip(train_x, train_y):
                d = int(digit)
                image = np.divide(image, 255)
                # forward prop
                image = image.reshape(784, 1)
                z1 = np.dot(self.w1, image) + self.b1
                h1 = self.sigmoid(z1)
                z2 = np.dot(self.w2, h1) + self.b2
                h2 = softmax(z2)
                predictionsVector = np.zeros(10)
                predictionsVector[d] = 1
                # back prop
                predictionsVector = np.asarray(predictionsVector).reshape(10, 1)
                dz2 = np.subtract(h2, predictionsVector)
                db2 = dz2
                dw2 = np.dot(dz2, np.transpose(h1))
                dh1 = np.dot(np.transpose(self.w2), dz2)
                dz1 = np.multiply(dh1, relu_derivative(z1))
                db1 = dz1
                dw1 = np.dot(dz1, np.transpose(image))
                # update params
                self.w1 = np.subtract(self.w1, np.multiply(dw1, LR))
                self.b1 = np.subtract(self.b1, np.multiply(db1, LR))
                self.w2 = np.subtract(self.w2, np.multiply(dw2, LR))
                self.b2 = np.subtract(self.b2, np.multiply(db2, LR))

    def predict(self, image):
        image = image.reshape(784, 1)
        z1 = np.dot(self.w1, image) + self.b1
        h1 = self.sigmoid(z1)
        z2 = np.dot(self.w2, h1) + self.b2
        h2 = softmax(z2)
        return np.argmax(h2)


    def test(self):
        #status("Test started")
        predictions = []
        for image in self.test_x:
            image = np.divide(image, 255)
            predictions.append(self.predict(image))
        return predictions


if __name__ == '__main__':
    #status("Started")
    nn = NN()
    nn.train()
    predictions = nn.test()
    np.savetxt("test_y", predictions, fmt='%d', delimiter='\n')
    #status("Finish")
