import gym
import numpy as np
import matplotlib.pyplot as plt

from qnet import QNet

N_ACTIONS = 4
N_STATES = 16
N_EPOCH = 1000
REWARD_DECAY = 0.9

def state_to_input(s):
    return np.eye(N_STATES)[s]

env = gym.make("FrozenLake-v1", render_mode="ansi")
env.reset()

qnet = QNet(N_STATES, N_ACTIONS)
epsilon = 0.9

reward_list = np.zeros(N_EPOCH)
for epoch in range(N_EPOCH):
    print(f"{epoch + 1} / {N_EPOCH}")

    total_reward = 0.0
    done = False
    s, _ = env.reset()
    while not done:
        epsilon = 1.0 / (epoch / 10 + 1.0)
        if np.random.rand() < epsilon:
            a = np.random.randint(N_ACTIONS)
            # print(f"random action : {a}")
        else:
            output = qnet.get_action_policy(state_to_input(s))
            a = np.argmax(output)
            # print(f"best action : {a}")

        s1, r, done, _, _ = env.step(a)
        # print(s, a, s1, r, done)

        qnet.update(state_to_input(s), a, state_to_input(s1), r)
        # print(qtable)

        s = s1
        total_reward += r
    
    reward_list[epoch] = total_reward

print(epsilon)    
print(qnet)
print(reward_list.mean())