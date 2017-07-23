import random
import numpy as np
import numpy.random as r

class linearMethod(object):
    def __init__(self,action):
        self.theta = np.zeros([8,1])
        #self.theta = np.random.randn(2, 1)
        self.y = 0.9
        self.a = 0.9
        self.action = action
        self.eps = 0.1

    def v(self,f):
        v = np.dot(f,self.theta)[0]

        return float(v)

    def greedy(self, f):
        action = self.action
        m = len(action)
        vv = []
        ss = 2*[0]
        for i in range(m):
            ss[0] = f[0] + action[i][0]
            ss[1] = f[1] + action[i][1]
            ff = self.f(ss)
            vv.append(self.v(ff))

        Vmax = max(vv)
        Vmaxes = []
        for i in range(len(vv)):
            if vv[i] == Vmax:
                Vmaxes.append(i)

        return random.choice(Vmaxes)


    def eGreedy(self, f):
        if random.random() < self.eps:
            return r.randint(4)
        else:
            return self.greedy(f)

    def MSVE(self,s,ss,r):
        y = self.y
        return (r + y*self.v(ss) - self.v(s))**2

    def f(self, s):
        f = list(s)
        f.append(s[0]*s[1])
        f.append(s[0]**2)
        f.append(s[1]**2)
        f.append((s[0]**2)*s[1])
        f.append(s[0]*(s[1]**2))
        f.append((s[0]**2)*(s[1]**2))

        return np.array(f)

    def update(self,s,ss,r):
        y = self.y
        a = self.a
        self.theta += a*(r+y*self.v(ss)-self.v(s))*np.array([s]).transpose()