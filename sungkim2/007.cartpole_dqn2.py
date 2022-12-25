import sys

import gym

from dqn import SimpleNet, DQN

N_EPOCH = 2000

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    dqn = DQN([0, 1], SimpleNet(4, 128, 2), SimpleNet(4, 128, 2))

    for epoch in range(N_EPOCH):
        s, _ = env.reset()
        done = False

        score = 0.0
        while not done:
            a = dqn.sample_action(s)
            s1, r, done, _, _ = env.step(a)
            dqn.store(s, a, s1, r, done)
            s = s1

            score += r
            if score > 10000.0:
                break

        dqn.train(epoch)

        print(f"{epoch + 1} {score} {dqn.epsilon}")
    env.close()

    env = gym.make('CartPole-v1', render_mode='human')
    s, _ = env.reset()
    done = False
    while not done:
        env.render()
        a = dqn.best_action(s)
        s_prime, r, done, _, _ = env.step(a)
        s = s_prime
    env.close()
        
