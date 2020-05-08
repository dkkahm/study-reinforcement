import numpy as np
import pandas as pd

class Agent:
  def __init__(self):
    self.states = {}

  def add_policy(self, state, action_p):
    self.states[state] = action_p

  def get_action(self, state):
    print(state)
    available_actions = [action for action in self.states[state]]
    print(available_actions)
    available_p = [self.states[state][action] for action in available_actions]
    print(available_p)
    next_action = np.random.choice(available_actions, p=available_p)
    print(next_action)
    return next_action

class MDP:
  def __init__(self, gamma=0.9):
    self.gamma = gamma
    self.states = {}
    self.state_names = set({})
    self.state_names_list = []

  def add_state(self, name, action_map = None, _pass = None):
    if name not in self.state_names:
      self.state_names_list.append(name)
      self.state_names.add(name)

    if _pass is None:
      self.states[name] = { '_value': 0.0, '_terminal': True, 'action_map': None, '_pass': None }
      return

    self.states[name] = { '_value': 0.0, '_terminal': False, '_action_map': action_map, '_pass': _pass }

    for action in _pass:
      total_p = 0.0
      pss = _pass[action]
      for next_state in pss:
        total_p += pss[next_state]
      if total_p != 1.0:
        raise ValueError('total p is not 1.0')

    self.states[name]['_pass'] = _pass

  def sample(self, start_state_name, agent):
    if start_state_name not in self.states:
      raise ValueError('Invalid start state')

    episode = []

    state_name = start_state_name
    while not self.states[state_name]['_terminal']:
      state = self.states[state_name]
      action = agent.get_action(state_name)
      print(action)
      print(self.states[state]['_pass'][action])
      available_states = [name for name in self.states[state]['_pass'][action]]
      available_pss = [self.states[state]['_pass'][action][name] for name in available_states]
      next_state = np.random.choice(available_states, p=available_pss)

      episode.append((state_name, action, self.states[state]['_action_map'][action]), next_state)

      state = next_state
    
    return episode

  def get_return_of_episode(self, episode):
    g = []
    gamma = self.gamma
    cg = 0.0
    for state_name, _, reward, _ in reversed(episode):
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
  mdp_student = MDP(gamma=1.0)

  mdp_student.add_state('Facebook', action_map={'Facebook': -1.0, 'Quit': 0.0 }, _pass={'Facebook': { 'Facebook': 1.0}, 'Quit': {'Class1': 1.0 }})
  mdp_student.add_state('Class1', action_map={'Facebook': -1.0, 'Study': -2.0 }, _pass={'Facebook': { 'Facebook': 1.0 }, 'Study': { 'Class2': 1.0 }})
  mdp_student.add_state('Class2', action_map={'Sleep': 0.0, 'Study': -2.0 }, _pass={'Sleep': {'Sleep': 1.0}, 'Study': {'Class3': 1.0}})
  mdp_student.add_state('Class3', action_map={'Study': 10.0, 'Pub': 1.0 }, _pass={'Study':{'Sleep': 1.0}, 'Pub': { 'Class1': 0.2, 'Class2': 0.4, 'Class3': 0.4 }})
  mdp_student.add_state('Sleep')
  # print(mdp_student.states)

  agent = Agent()
  agent.add_policy('Facebook', {'Facebook': 0.5, 'Quit': 0.5 })
  agent.add_policy('Class1', {'Facebook': 0.5, 'Study': 0.5 })
  agent.add_policy('Class2', {'Study': 0.5, 'Sleep': 0.5 })
  agent.add_policy('Class3', {'Study': 0.5, 'Pub': 0.5 })

  # mdp_student.reset_value()

  for n in range(1):
    episode = mdp_student.sample('Class1', agent)
    # print(episode)
    # g = mdp_student.get_return_of_episode(episode)
    # print(g)
    mdp_student.update_value(episode, n + 1)
  print(mdp_student.states)

  # pss = mdp_student.get_transition_matrix()
  # print(pss)