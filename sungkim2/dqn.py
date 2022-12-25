import collections
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer():
    def __init__(self, buffer_limit = 50000):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class DQN:
    def __init__(self, actions, main_net, target_net, learning_rate=0.0005, gamma=0.98, buffer_limit=50000, batch_size=32, train_limit=2000, net_copy_interval = 20):
        self.actions = actions

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.buffer_limit = buffer_limit
        self.batch_size = batch_size
        self.train_limit = train_limit
        self.net_copy_interval = net_copy_interval

        self.main_net = main_net
        self.target_net = target_net
        self.target_net.load_state_dict(self.main_net.state_dict())

        self.memory = ReplayBuffer(buffer_limit)

        self.optimizer = optim.Adam(self.main_net.parameters(), lr=learning_rate)

        self.epsilon = 0.08

    def sample_action(self, obs):
        out = self.main_net.forward(torch.from_numpy(obs).float())
        coin = random.random()
        if coin < self.epsilon:
            return random.choice(self.actions)
        else : 
            return out.argmax().item()

    def best_action(self, obs):
        out = self.main_net.forward(torch.from_numpy(obs).float())
        return out.argmax().item()

    def store(self, s, a, s1, r, done):
        self.memory.put((s, a, r/100.0, s1, 0.0 if done else 1.0))

    def train(self, epoch):
        self.epsilon = max(0.01, 0.08 - 0.01*(epoch/200))

        if self.memory.size() > self.train_limit:
            for i in range(10):
                s, a, r, s_prime, done_mask = self.memory.sample(self.batch_size)

                q_out = self.main_net(s)
                q_a = q_out.gather(1,a)
                max_q_prime = self.target_net(s_prime).max(1)[0].unsqueeze(1)
                target = r + self.gamma * max_q_prime * done_mask
                loss = F.smooth_l1_loss(q_a, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
        if epoch % self.net_copy_interval == 0:
            self.target_net.load_state_dict(self.main_net.state_dict())
