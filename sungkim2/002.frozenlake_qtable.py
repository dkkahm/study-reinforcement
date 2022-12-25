import gym
import numpy as np
import matplotlib.pyplot as plt

from qtable import QTable

N_ACTIONS = 4
N_STATES = 16
N_EPOCH = 2000

env = gym.make("FrozenLake-v1", render_mode="ansi")
env.reset()

qtable = QTable(N_STATES, N_ACTIONS)
epsilon = 0.9

reward_list = np.zeros(N_EPOCH)
for epoch in range(N_EPOCH):
    print(f"{epoch + 1} / {N_EPOCH}")

    total_reward = 0.0
    done = False
    s, _ = env.reset()
    while not done:
        epsilon = 1.0 / (epoch / 100 + 1.0)
        if np.random.rand() < epsilon:
            a = np.random.randint(N_ACTIONS)
            # print(f"random action : {a}")
        else:
            a = qtable.get_best_action(s)
            # print(f"best action : {a}")

        s1, r, done, _, _ = env.step(a)
        # print(s, a, s1, r, done)

        qtable.update(s, a, r, s1)
        # print(qtable)

        s = s1
        total_reward += r
    
    reward_list[epoch] = total_reward

print(epsilon)    
print(qtable)
print(reward_list.mean())
plt.plot(reward_list)
plt.show()
