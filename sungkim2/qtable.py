import os, sys
import numpy as np

class QTable:
    def __init__(self, num_state, num_action, learning_rate=0.1, reward_decay=0.9):
        self.num_state = num_state
        self.num_action = num_action
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.qtable = np.zeros((num_state, num_action))

    def __repr__(self):
        return f"{self.qtable}"

    def get_best_action(self, state):
        return np.argmax(self.qtable[state])

    def update(self, s, a, r, s1):
        self.qtable[s, a] = (1.0 - self.learning_rate) * self.qtable[s, a] + \
                            self.learning_rate * (r + self.reward_decay * np.max(self.qtable[s1]))
