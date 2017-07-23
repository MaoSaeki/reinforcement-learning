import random
import numpy as np
import numpy.random as r

class Sigmoid(object):
    @staticmethod
    def f(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def prime(z):
        sigma = Sigmoid.f(z)
        return sigma * (1 - sigma)

class QNeuralNet():
    def __init__(self, size, eta):
        self.size = size
        self.layerN = len(size)
        self.biases = [np.random.randn(x, 1) for x in size[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(size[1:], size[:-1])]
        self.eta = eta
        self.eps = 0.1
        self.y = 0.9

    def update(self, action,s,ss,r):
        s = np.array([s])
        ss = np.array([ss])
        A = []
        Z = []
        a = s.transpose()


        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            Z.append(z)
            a = Sigmoid.f(z)
            A.append(a)

        y = np.zeros([4, 1])
        y[np.argmax(a), 0] = -r

        Qs = self.feedForward(s)
        Qmax = max(self.feedForward(ss))
        target = y + self.y * Qmax
        error = [np.zeros(b.shape) for b in self.biases]
        error[-1] = (target - Qs[action]) * Sigmoid.prime(Z[-1]) *0.01
        for l in range(2, self.layerN):
            error[-l] = np.dot(self.weights[-l + 1].transpose(), error[-l + 1]) * Sigmoid.prime(Z[-l])

        deltaB = error
        deltaW = [a * e for a, e in zip(A, error)]
        self.biases = [b - self.eta * dB for b, dB in zip(self.biases, deltaB)]
        self.weights = [w - self.eta * dW for w, dW in zip(self.weights, deltaW)]

    def a(self, s):
        Qs = self.feedForward(s)
        #print(Qs)
        Qmax = max(Qs)
        Qmaxes = []
        for i in range(len(Qs)):
            if Qs[i] == Qmax:
                Qmaxes.append(i)

        return random.choice(Qmaxes)

    def aEps(self, s):
        if random.random() < self.eps:
            return r.randint(4)
        else:
            s=np.array([s])
            return self.a(s)

    def feedForward(self, s):
        a = s.transpose()
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a) + b
            a = Sigmoid.f(z)
        # print(a)
        return a

