import numpy as np

class GridWorld():
    def __init__(self,action,row,column):
        self.startposition = [0, 3]
        self.position = self.startposition
        self.windyRow = {3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 1}
        self.goal = [7, 3]
        self.action = action
        self.row = row
        self.column = column

    def takeAction(self, a, s):
        self.position[0] += self.action[a][0]
        self.position[1] += self.action[a][1]
        try:
            self.position[1] += self.windyRow[self.position[0]]
        except:
            pass

        if self.position[0] < 0:
            self.position[0] = 0
        if self.position[0] > self.column - 1:
            self.position[0] = self.column - 1
        if self.position[1] < 0:
            self.position[1] = 0
        if self.position[1] > self.row - 1:
            self.position[1] = self.row - 1

        return np.array(self.position), self.reward()

    def reward(self):
        if self.Goal():
            return 0
        else:
            return -1

    def Goal(self):
        if self.position == self.goal:
            return True
        else:
            return False

    def reset(self):
        self.position = [0, 3]