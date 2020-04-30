import random


class ActionSpace:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return int(random.random() * self.n)
