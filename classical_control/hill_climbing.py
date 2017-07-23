import numpy as np

class HillClimbing():
    def run_episode(self, env, parameters):
        total_reward = 0
        observation = env.reset()
        for t in range(200):
            env.render()
            action = 0 if np.dot(parameters, observation) < 0 else 1
            observation, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                print("Episode finished afetr {} timesteps".format(t + 1))
                break

        return total_reward

