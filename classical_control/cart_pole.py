import gym
import q_learning
import numpy as np
from collections import deque

class CartPole():
    def __init__(self, env, model):
        self.env = env
        self.model = model

    def run_episode(self, render=False):
        total_reward = 0
        state = self.env.reset()
        for t in range(200):
            if render:
                self.env.render()

            action = self.model.get_action(state)
            next_state, reward, done, info = self.env.step(action)

            self.model.stock_exp(state, action, reward, next_state, done)
            self.model.train()

            total_reward += reward
            state = next_state
            if done:
                if False:
                    print("Episode finished afetr {} timesteps".format(t + 1))
                break

        return total_reward, t+1

    def run_epoch(self, n_episode, render=False):
        rewards = deque()
        total_step = 0
        for i in range(n_episode):
            total_reward, t = self.run_episode(render)
            rewards.append(total_reward)
            total_step += t

            reward_av = np.mean(rewards)
            print("total reward: %3d, average reward: %.3f,  episode: %d" % (total_reward, reward_av, i + 1))
            if len(rewards) >= 100:
                if reward_av >= -110:
                    print("solved in {} episodes".format(i+1))
                    return
                else:
                    rewards.popleft()
        print("could not be solved after {} episodes".format(n_episode))

def evaluate_average_to_goal():
    tto = []
    gym_env = gym.make('MountainCar-v0')
    n_act = gym_env.action_space.n
    n_obs = gym_env.observation_space.shape[0]

    bonuses = [0, 1,10,100,1000]
    for b in bonuses:
        print("exploration bonus = {}".format(b))
        for i in range(100):
            model = q_learning.Q_learning(n_obs, n_act)
            model.ExplorationBonus = True
            model.exp_bonus = 1
            env = CartPole(gym_env, model)
            for i in range(1000):
                total_reward, time_step = env.run_episode(render=False)
                if time_step < 200:
                    tto.append(i)
                    break
        print("average {} epoch to first success".format(np.mean(tto)))



def single():
    # gym_env = gym.make('CartPole-v0')
    gym_env = gym.make('MountainCar-v0')
    n_act = gym_env.action_space.n
    n_obs = gym_env.observation_space.shape[0]
    model = q_learning.Q_learning(n_obs, n_act)

    env = CartPole(gym_env, model)
    env.run_epoch(10000, False)

if __name__=='__main__':
    single()










