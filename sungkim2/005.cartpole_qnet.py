import gym
import numpy as np
import matplotlib.pyplot as plt

from qnet import QNet

SIZE_STATES = 4
N_ACTIONS = 2
N_EPOCH = 1000

# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1", render_mode=None)

qnet = QNet(SIZE_STATES, N_ACTIONS)
epsilon = 0.9

reward_list = np.zeros(N_EPOCH)
for epoch in range(N_EPOCH):
    # print(f"{epoch + 1} / {N_EPOCH}")
    s, _ = env.reset()

    env.render()

    total_reward = 0
    done = False
    while not done:
        epsilon = 1.0 / (epoch / 10 + 1.0)
        if np.random.rand() < epsilon:
            a = np.random.randint(N_ACTIONS)
            # print(f"random action : {a}")
        else:
            output = qnet.get_action_policy(s)
            a = np.argmax(output)

        s1, r, done, _, _ = env.step(a)
        # print(s, a, s1, r, done)

        qnet.update(s, a, s1, r)

        total_reward += r

        env.render()

        s = s1
    reward_list[epoch] = total_reward

    print(f"{epoch + 1} / {N_EPOCH} {total_reward}")

print(reward_list.mean())
plt.plot(reward_list)
plt.show()
