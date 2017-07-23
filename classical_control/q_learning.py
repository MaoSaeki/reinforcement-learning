import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

import copy
from collections import deque

class MyChain(Chain):
    def __init__(self, n_obs, n_act):
        super(MyChain, self).__init__(
            L1=L.Linear(n_obs, 100),
            L2=L.Linear(100, 100),
            L3=L.Linear(100, 100),
            L4 = L.Linear(100, n_act, initialW=np.zeros((n_act, 100), dtype=np.float32))
        )

    def __call__(self, x):
        h = F.leaky_relu(self.L1(x))
        h = F.leaky_relu(self.L2(h))
        h = F.leaky_relu(self.L3(h))
        h = F.leaky_relu(self.L4(h))
        return h


class Q_learning():
    def __init__(self, n_obs, n_act):
        self.MemoryShuffle = True
        self.ExplorationBonus = True

        np.random.seed(100)
        self.chain = MyChain(n_obs, n_act)
        self.target_chain = copy.deepcopy(self.chain)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.chain)

        self.memory = deque()
        self.memsize = 1000
        self.batchsize = 100
        self.gamma = 0.99
        self.epsilon = 1
        self.eps_decay = 0.005

        self.step = 0
        self.target_update_freq = 20
        self.train_freq = 10
        self.eps_min = 0
        self.exploration = 1000
        self.exploration_freq = 40

        self.explored_range = np.zeros([n_obs, 2])
        self.exp_bonus = 10
        self.n_obs = n_obs

    def stock_exp(self, s, a, r, ss, terminal):
        self.memory.append([s, a, r, ss, terminal])
        if len(self.memory) > self.memsize:
            self.memory.popleft()

    def get_action(self, s):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, 2)
        else:
            Q = self.chain(np.array([s], dtype=np.float32)).data
            a = np.argmax(Q)
            return a

    def epsilon_decay(self):
        if self.epsilon >= self.eps_min and self.exploration < self.step:
            self.epsilon -= self.eps_decay

    def shuffle_memory(self):
        mem = np.array(self.memory)
        return np.random.permutation(mem)

    def exploration_bonus(self, QQ):
        N = len(QQ)
        bonus = np.zeros(N)
        if self.ExplorationBonus:
            for i in range(N):
                for n in range(self.n_obs):
                    if QQ[i][n] < self.explored_range[n][0]:
                        bonus[i] += self.exp_bonus
                        self.explored_range[n][0] = QQ[i][n]
                    elif self.explored_range[n][1] < QQ[i][n]:
                        bonus[i] += self.exp_bonus
                        self.explored_range[n][1] = QQ[i][n]

        return bonus

    def back_prop(self, s, a, r, ss, terminal):
        Q = self.chain(s)
        QQ = self.target_chain(ss)
        QQ_max = np.array([np.max(qq) for qq in QQ.data])
        target = copy.deepcopy(Q.data)
        exp_bonus = self.exploration_bonus(QQ.data)
        for i in range(len(s)):
            target[i][a[i]] = r[i] + self.gamma * QQ_max[i] * (not terminal[i]) + exp_bonus[i]

        self.chain.cleargrads()
        loss = F.mean_squared_error(Q, Variable(target))
        loss.backward()
        self.optimizer.update()

    def parse_batch(self, batch):
        s, a, r, ss, terminal = [], [], [], [], []
        for b in batch:
            s.append(b[0])
            a.append(b[1])
            r.append(b[2])
            ss.append(b[3])
            terminal.append(b[4])

        s = np.array(s, dtype=np.float32)
        a = np.array(a, dtype=np.int8)
        r = np.array(r, dtype=np.float32)
        ss = np.array(ss, dtype=np.float32)
        terminal = np.array(terminal, dtype=np.bool)

        return s, a, r, ss, terminal

    def experience_replay(self):
        mem = self.shuffle_memory()
        N = len(mem)
        for i in range(0, N, self.batchsize):
            batch = mem[i:i+self.batchsize]
            s, a, r, ss, terminal = self.parse_batch(batch)
            self.back_prop(s, a, r, ss, terminal)

    def train(self):
        if len(self.memory) >= self.memsize:
            if self.step % self.train_freq == 0:
                self.experience_replay()
                self.epsilon_decay()
            if self.step % self.target_update_freq == 0:
                self.target_chain = copy.deepcopy(self.chain)
            # if self.step % self.exploration_freq == 0:
            #     self.epsilon += 0.1

        self.step += 1