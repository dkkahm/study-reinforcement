import numpy as np
import pandas as pd

class MP:
  def __init__(self):
    self.states = {}
    self.state_names = set({})
    self.state_names_list = []

  def add_state(self, name, pss = None):
    if name not in self.state_names:
      self.state_names_list.append(name)
      self.state_names.add(name)

    if pss is None:
      self.states[name] = { '_terminal': True, '_pss': { name: 1.0 } }
      return

    self.states[name] = { '_terminal': False }

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

if __name__ == '__main__':
  mp_student = MP()

  mp_student.add_state('Facebook', {'Facebook': 0.9, 'Class1': 0.1})
  mp_student.add_state('Class1', {'Facebook': 0.5, 'Class2': 0.5})
  mp_student.add_state('Class2', {'Sleep': 0.2, 'Class3': 0.8})
  mp_student.add_state('Class3', {'Pass': 0.6, 'Pub': 0.4})
  mp_student.add_state('Pub', {'Class1': 0.2, 'Class2': 0.4, 'Class3': 0.4})
  mp_student.add_state('Pass', {'Sleep': 1.0})
  mp_student.add_state('Sleep')
  # print(mp_student.get_states())

  # episode = mp_student.sample('Class1')
  # print(episode)

  # pss = mp_student.get_transition_matrix()
  # print(pss)