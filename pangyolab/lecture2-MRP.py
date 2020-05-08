import numpy as np
import pandas as pd

class MRP:
  def __init__(self, gamma):
    self.gamma = gamma
    self.states = {}
    self.state_names = set({})
    self.state_names_list = []

  def add_state(self, name, reward = 0.0, pss = None):
    if name not in self.state_names:
      self.state_names_list.append(name)
      self.state_names.add(name)

    if pss is None:
      self.states[name] = { '_value': 0.0, '_terminal': True, '_reward': reward, '_pss': { name: 1.0 } }
      return

    self.states[name] = { '_value': 0.0, '_terminal': False, '_reward': reward }

    total_p = 0.0
    for next_state in pss:
      total_p += pss[next_state]
    if total_p != 1.0:
      raise ValueError('total p is not 1.0')

    self.states[name]['_pss'] = pss

  def sample(self, start_state):
    if start_state not in self.states:
      raise ValueError('Invalid start state')

    episode = [start_state]

    state = start_state
    while not self.states[state]['_terminal']:
      available_states = [name for name in self.states[state]['_pss']]
      available_pss = [self.states[state]['_pss'][name] for name in available_states]
      next_state = np.random.choice(available_states, p=available_pss)
      episode.append(next_state)
      state = next_state
    
    return episode

  def get_transition_matrix(self):
    state_names_list = self.state_names_list
    transition_matrix = np.zeros((len(state_names_list), len(state_names_list)))

    for state_name in self.states:
      state_index = state_names_list.index(state_name)
      state = self.states[state_name]

      for next_state_name in state['_pss']:
        next_state_index = state_names_list.index(next_state_name)

        transition_matrix[state_index, next_state_index] = state['_pss'][next_state_name]

    return pd.DataFrame(transition_matrix, index=state_names_list, columns=state_names_list)

  def get_return_of_episode(self, episode):
    g = []
    # cg = 0.0
    # gamma = 1.0
    # x = 0.0

    # ??? why not this ???
    # gamma = self.gamma
    # cg = 0.0
    # for i in range(len(episode) - 1, 0, -1):
    #   state_name = episode[i - 1]
    #   next_state_name = episode[i]
    #   next_state = self.states[next_state_name]
    #   reward = next_state['_reward']

    #   cg = cg * gamma + reward
    #   g = [(state_name, cg)] + g

    gamma = self.gamma
    cg = 0.0
    for state_name in reversed(episode):
      # print(state_name)
      state = self.states[state_name]
      reward = state['_reward']

      cg = cg * gamma + reward
      g = [(state_name, cg)] + g

    return g

  def reset_value(self):
    for name in self.states:
      if not self.states[name]['_terminal']:
        self.states[name]['_value'] = 0.0

  def update_value(self, episode, n):
    if n < 1:
      raise ValueError('n must greater than 0')

    g = self.get_return_of_episode(episode)

    for state_name, g_of_state in g:
      state = self.states[state_name]
      value = state['_value']
      new_value = g_of_state / n + (n - 1) / n * value
      state['_value'] = new_value

if __name__ == '__main__':
  mrp_student = MRP(gamma=1.0)

  mrp_student.add_state('Facebook', reward=-1.0, pss={'Facebook': 0.9, 'Class1': 0.1})
  mrp_student.add_state('Class1', reward=-2.0, pss={'Facebook': 0.5, 'Class2': 0.5})
  mrp_student.add_state('Class2', reward=-2.0, pss={'Sleep': 0.2, 'Class3': 0.8})
  mrp_student.add_state('Class3', reward=-2.0, pss={'Pass': 0.6, 'Pub': 0.4})
  mrp_student.add_state('Pub', reward=1.0, pss={'Class1': 0.2, 'Class2': 0.4, 'Class3': 0.4})
  mrp_student.add_state('Pass', reward=10.0, pss={'Sleep': 1.0})
  mrp_student.add_state('Sleep')
  # print(mrp_student.states)

  # mrp_student.reset_value()

  for n in range(100000):
    episode = mrp_student.sample('Class1')
    # print(episode)
    # g = mrp_student.get_return_of_episode(episode)
    # print(g)
    mrp_student.update_value(episode, n + 1)
  print(mrp_student.states)

  # pss = mrp_student.get_transition_matrix()
  # print(pss)