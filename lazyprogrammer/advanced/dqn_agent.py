import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
from collections import deque

class DQNAgent:
  def __init__(self, env, gamma = 0.9, replay_memory_size=2000, start_train_size=1000, batch_size=64, learning_rate=0.001, epsilon_decay=0.999, epsilon_min=0.01):
    self.env = env
    self.gamma = gamma
    self.replay_memory_size = replay_memory_size
    self.start_train_size = start_train_size
    self.batch_size = batch_size
    self.learning_rate = learning_rate

    self.epsilon = 1.0
    self.epsilon_decay = epsilon_decay
    self.epsilon_min = epsilon_min

    self.behavior_model = self._build_model()
    self.target_model = self._build_model()
    self._update_model()

    self.replay_memory = deque(maxlen=self.replay_memory_size)

    # print("gamma=", self.gamma,
    #   "replay_memory_size=", self.replay_memory_size,
    #   "start_train_size=", self.start_train_size,
    #   "batch_size=", self.batch_size,
    #   "learning_rate=", self.learning_rate,
    #   "epsilon_decay=", self.epsilon_decay,
    #   "epsilon_min=", self.epsilon_min)

  def get_action_train(self, s):
    if np.random.rand() <= self.epsilon:
      return self.env.action_space.sample()

    q = self.behavior_model.predict(s.reshape(1, -1))

    return np.argmax(q[0])

  def get_action(self, s):
    q = self.behavior_model.predict(s.reshape(1, -1))
    return np.argmax(q[0])    

  def train_episode(self, render=False):
    t = 0
    total_rewards = 0.
    done = False
    s = self.env.reset()
    while not done:
      t += 1

      if render:
        self.env.render()

      a = self.get_action_train(s)
      s1, r, done, _ = self.env.step(a)

      r = self._adjust_reward(t, s, a, r, s1, done)
      # print("    ", t, s, a, s1, r, done, len(self.replay_memory))
      total_rewards += r

      self.replay_memory.append((s, a, r, s1, done))

      self._train(t)

      s = s1

      if self._early_break(t, s, a, r, s1, done):
        break

    self._update_model()

    return t, total_rewards

  def run_episode(self, render=False, env=None):
    if env is None:
      env = self.env

    t = 0
    total_rewards = 0.
    done = False
    s = env.reset()
    while not done:
      t += 1

      if render:
        env.render()

      a = self.get_action(s)
      s1, r, done, _ = env.step(a)

      r = self._adjust_reward(t, s, a, r, s1, done)
      total_rewards += r

      s = s1

      if self._early_break(t, s, a, r, s1, done):
        break

    return t, total_rewards

  def _train(self, episode):
    if len(self.replay_memory) < self.start_train_size:
      return

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

    samples = random.sample(self.replay_memory, self.batch_size)

    s_batch = np.empty((self.batch_size, self.env.observation_space.shape[0]))
    s1_batch = np.empty((self.batch_size, self.env.observation_space.shape[0]))
    a_batch = np.empty((self.batch_size,))
    r_batch = np.empty((self.batch_size,))
    done_batch = np.empty((self.batch_size,))

    rmax_batch = np.empty((self.batch_size,))

    for i, (s, a, r, s1, done) in enumerate(samples):
      s_batch[i, :] = s
      s1_batch[i, :] = s1
      a_batch[i] = a
      r_batch[i] = r
      done_batch[i] = 1. if done else 0.

    q_batch = self.behavior_model.predict(s_batch)
    q1_batch = self.target_model.predict(s1_batch)
    rmax_batch = r_batch + self.gamma * q1_batch.max(axis=1)

    # print("s_batch=", s_batch)
    # print("s1_batch=", s1_batch)
    # print("a_batch=", a_batch)
    # print("r_batch=", r_batch)
    # print("done_batch=", done_batch)
    # print("q_batch=", q_batch)
    # print("q1_batch=", q1_batch)
    # print("rmax_batch=", rmax_batch)

    for a in range(self.env.action_space.n):
      a_mask = a_batch == a
      q_batch[a_mask, a] = (r_batch * done_batch + rmax_batch * (1.0 - done_batch))[a_mask]
    # print("target_batch=", q_batch)

    self.behavior_model.fit(s_batch, q_batch, batch_size=self.batch_size, epochs=1, verbose=0)


    # for i, (s, a, r, s1, done) in enumerate(samples):
    #   q = self.behavior_model.predict(s.reshape(1, -1)).flatten()
    #   q1 = self.target_model.predict(s1.reshape(1, -1)).flatten()

    #   if done:
    #     target = r
    #   else:
    #     target = r + self.gamma * np.max(q1)

    #   q[a] = target

    #   s_batch[i, :] = s
    #   q_batch[i, :] = q

    # self.behavior_model.fit(s_batch, q_batch, batch_size=self.batch_size, epochs=1, verbose=0)

  def _build_model(self):
    model = Sequential()
    model.add(Dense(24, input_shape=self.env.observation_space.shape, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(self.env.action_space.n, activation='linear', kernel_initializer='he_uniform'))
    model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
    return model

  def _update_model(self):
    self.target_model.set_weights(self.behavior_model.get_weights())

  def _adjust_reward(self, t, s, a, r, s1, done):
    return r

  def _early_break(self, t, s, a, r, s1, done):
    return False
