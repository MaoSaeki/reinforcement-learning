import csv

import numpy as np
from LinearMethod import linearMethod
from Q import Q
from Sarsa import Sarsa
from neaural_net import QNeuralNet

from gridwold.GridWorld import GridWorld

action = [[0, 1], [1, 0], [0, -1], [-1, 0]]
row = 7
column = 10

def solveSarsa(episode, epoch):

    averageStep = np.zeros(episode)
    for k in range(epoch):
        W = GridWorld(action, row, column)
        Sar = Sarsa(action, row, column)
        for i in range(episode):
            t = 0
            W.reset()
            s = [0, 3]
            A = Sar.aEps(s)
            while not W.Goal():
                # if i == epoch - 1:
                # print("s:", s)
                ss, r = W.takeAction(A, s)
                AA = Sar.aEps(ss)
                Sar.update(s, ss, A, AA, r)
                s = ss
                A = AA
                t += 1
            averageStep[i] += t
    averageStep/=epoch
    f = open('Sarsa.csv', 'w')
    dataWriter = csv.writer(f)
    for r in range(episode):
        dataWriter.writerow([r + 1, averageStep[r]])
    print("saved")
    f.close()

def solveQ(episode, epoch):

    averageStep = np.zeros(episode)
    for k in range(epoch):
        print("epoch ",k)
        W = GridWorld(action, row, column)
        q = Q(action, row, column)
        for i in range(episode):
            t = 0
            W.reset()
            s = [0, 3]
            while not W.Goal():
                A = q.aEps(s)
                ss, r = W.takeAction(A, s)
                q.update(s, ss, A, r)
                s=ss
                t += 1
            averageStep[i] += t
    averageStep/=epoch
    f = open('Q2.csv', 'w')
    dataWriter = csv.writer(f)
    for r in range(episode):
        dataWriter.writerow([r + 1, averageStep[r]])
    print("saved")
    f.close()

def solveQNN():
    W = GridWorld(action, row, column)
    qnn = QNeuralNet([2, 20, 10, 4], 0.2)
    epoch = 1
    for i in range(epoch):

        for k in range(100):
            t = 0
            W.reset()
            s = np.array([0, 3])
            while not W.Goal():
                print(s)
                a = qnn.aEps(s)
                #print(a)
                ss, r = W.takeAction(a, s)
                qnn.update(a,s,ss,r)
                s = ss
                t += 1
            print("time step:", t)

def soveLinearMethod(episode, epoch):

    for k in range(epoch):
        W = GridWorld(action, row, column)
        LM = linearMethod(action)
        for i in range(episode):
            t = 0
            W.reset()
            s = np.array([0, 3])
            while not W.Goal():
                f = LM.f(s)
                a = LM.e_greedy(f)
                ss, r = W.takeAction(a, s)
                ff = LM.f(ss)
                msve = LM.MSVE(f,ff,r)
                print(msve)
                LM.update(f,ff,r)
                s = ss
                t += 1


solveQ(200, 100)
#solveQNN()
#soveLinearMethod(200,2)
