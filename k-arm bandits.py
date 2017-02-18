import csv
import numpy as np


# k-arm bandit
class KArmBandits:
    def __init__(self, k):
        self.k = k
        self.a = np.random.randn(k)

    def select_band(self, action):
        return self.a[action] + np.random.randn()


# action毎の期待値Qの保持、更新、actionの選択を行う
class Estimate:
    def __init__(self, k, eps, intial_value):
        self.k = k
        self.eps = eps
        self.Q = np.zeros(k)
        self.Q.fill(intial_value)
        self.counter = np.zeros(k)

    # greedy methodによりactionを選択
    def greedy(self):
        return np.argmax(self.Q)

    # ε-greedy methodによりactionを選択
    def e_greedy(self):
        if np.random.rand() < self.eps:
            return np.random.randint(self.k)
        else:
            return self.greedy()

    # action毎の期待値を更新
    def update(self, action, reward):
        self.Q[action] = (self.Q[action] * self.counter[action] + reward) / (self.counter[action] + 1)
        self.counter[action] += 1

    # ε-greedyのεを更新
    def update_eps(self, eps):
        self.eps = eps


# k-arm bandit問題をoptomistic_greedyで行い結果を保存
def run_karm_bandit_optomistic_greedy(k, step, episode):
    print("learning...")
    avr_score = np.zeros(step)

    for i in range(1, episode + 1):
        band = KArmBandits(k)
        eps = 0.01
        q = Estimate(k, eps, 5)
        score = 0
        for t in range(step):
            # action:選択するarmed banditの番号
            action = q.e_greedy()
            reward = band.select_band(action)
            q.update(action, reward)
            score += reward
            avr = score / (t + 1)
            avr_score[t] = (avr_score[t] * (i - 1) + avr) / i
            if t == 200:
                eps = 0.01
                q.update_eps(eps)
        progress = int(episode / 10)
        if i % progress == 0:
            print(str(i / progress * 10) + "%")

    print("saving as 'k-arm bandits optomistic greedy update.csv'")
    file = open('k-arm bandits optomistic greedy update.csv', 'w')
    data_writer = csv.writer(file)
    for row in range(step):
        data_writer.writerow([row + 1, avr_score[row]])

    file.close()


run_karm_bandit_optomistic_greedy(10, 1000, 2000)
