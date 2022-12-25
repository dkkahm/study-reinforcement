import os, sys
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = self.fc1(x)
        return out


def tensor_to_array(t):
    return t.cpu().detach().numpy()


def array_to_tensor(a):
    return torch.tensor(a.astype(np.float32)).to(DEVICE)


class QNet:
    def __init__(self, num_state, num_action, learning_rate=0.1, reward_decay=0.9):
        self.num_state = num_state
        self.num_action = num_action
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay

        model = FeedforwardNeuralNetModel(num_state, num_action)
        self.model = model.to(DEVICE)

        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
       
        
    def __repr__(self):
        s = ""
        for param in self.model.parameters():
            if len(s) > 0:
                s += "\n"
            s += str(param.data)
        return s


    def get_action_policy(self, state):
        self.model.eval()
        with torch.no_grad():
            action_policy = self.model(array_to_tensor(state.reshape(1, -1)))
        return tensor_to_array(action_policy)[0]


    def update(self, s, a, s1, r):
        target_action_policy = self.get_action_policy(s)
        s1_action_policy = self.get_action_policy(s1)
        
        target_action_policy[a] = r + self.reward_decay * np.max(s1_action_policy)

        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(array_to_tensor(s.reshape(1, -1)))

        loss = self.loss_func(output, array_to_tensor(target_action_policy.reshape(1, -1)))
        loss.backward()
        self.optimizer.step()
