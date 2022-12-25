import gym
import numpy as np
import matplotlib.pyplot as plt

N_EPOCH = 1000

# env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1", render_mode=None)

reward_list = np.zeros(N_EPOCH)
for epoch in range(N_EPOCH):
    # print(f"{epoch + 1} / {N_EPOCH}")
    s, _ = env.reset()

    env.render()

    total_reward = 0
    done = False
    while not done:
        a = env.action_space.sample()
        s1, r, done, _, _ = env.step(a)
        # print(s, a, s1, r, done)

        total_reward += r

        env.render()

        s = s1
    reward_list[epoch] = total_reward

    print(f"{epoch + 1} / {N_EPOCH} {total_reward}")

print(reward_list.mean())
plt.plot(reward_list)
plt.show()
