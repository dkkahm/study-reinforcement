# https://wegonnamakeit.tistory.com/59
import sys

import gym
import collections
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32

class ReplayBuffer():
    def __init__(self):
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

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0,1)
        else : 
            return out.argmax().item()

def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        # print("q_out", q_out)
        q_a = q_out.gather(1,a)
        # print("q_a", q_a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        # target_out = q_target(s_prime)
        # print("1", target_out)
        # target_out = target_out.max(1)
        # print("2", target_out)
        # target_out = target_out[0]
        # print("3", target_out)
        # max_q_prime = target_out.unsqueeze(1)
        # print("max_q_prime", max_q_prime)

        target = r + gamma * max_q_prime * done_mask
        # print("target", target)
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # sys.exit(1)


def main():
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(1000):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        s, _ = env.reset()
        done = False
        scroe = 0.0
        while not done:
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)      
            s_prime, r, done, info, _ = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break

            if score > 10000:
                break
            
        if memory.size()>2000:
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score, memory.size(), epsilon*100))
            score = 0.0
    env.close()

    env = gym.make('CartPole-v1', render_mode="human")

    s, _ = env.reset()
    done = False

    while not done:
        env.render()
        a = q.sample_action(torch.from_numpy(s).float(), 0.0)
        s_prime, r, done, info, _ = env.step(a)
        s = s_prime

    env.close()

if __name__ == '__main__':
    main()
