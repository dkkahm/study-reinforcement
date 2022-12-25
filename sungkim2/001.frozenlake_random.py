import gym
import numpy as np

env = gym.make("FrozenLake-v1", render_mode="ansi")
env.reset()

x = env.render()
print(x)

done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, _, _ = env.step(action)
    print(observation, reward, done)

    x = env.render()
    print(x)
