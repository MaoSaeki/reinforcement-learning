import numpy as np
import random
import numpy.random as r

class Q():
    def __init__(self,action,row,column):
        self.Q = [[np.zeros(len(action)) for r in range(row)] for c in range(column)]
        self.eps = 0.1
        self.alpha = 0.5
        self.y = 0.9

    def a(self, s):
        Qs = self.Qs(s)
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
            return self.a(s)

    def Qs(self, s):
        return self.Q[s[0]][s[1]]

    def update(self, s, ss, a, R):
        alpha = self.alpha
        y = self.y
        Q = self.Qs(s)[a]
        Qmax = max(self.Qs(ss))
        self.Q[s[0]][s[1]][a] = Q + alpha * (R + y * Qmax - Q)