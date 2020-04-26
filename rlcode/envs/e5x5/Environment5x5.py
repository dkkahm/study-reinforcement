from .. import actions
from ..ActionSpace import ActionSpace
from ..ObservationSpace import ObservationSpace

class Environment5x5:
  def __init__(self):
    self.observation_space = ObservationSpace(25)
    self.action_space = ActionSpace(4)
    self.reset()

  def reset(self):
    self.s = 0
    self.last_a = None
    self.done = False
    return self.s

  def reset_state(self, s):
    self.last_a = None
    self.s = s
    self.done = Environment5x5._is_termial(s)

  def step(self, a):
    if self.done:
      return (self.s, 0, self.done, None)

    srow = self.s // 5
    scol = self.s % 5
    # print(self.s, action_name(a), (srow, scol), "=>", end="")

    if a == actions.LEFT:
      if scol > 0:
        scol -= 1
    elif a == actions.DOWN:
      if srow < 4:
        srow += 1
    elif a == actions.RIGHT:
      if scol < 4:
        scol += 1
    elif a == actions.UP:
      if srow > 0:
        srow -= 1
    else:
      raise ValueError
    # print((srow, scol))

    self.last_a = a
    self.s = srow * 5 + scol
    self.done = Environment5x5._is_termial(self.s)

    return (self.s, Environment5x5._get_reward(self.s), self.done, None)

  def render(self):
    if self.last_a is not None:
      print(f"({actions.action_name(self.last_a)})")
            
    srow = self.s // 5
    scol = self.s % 5
        
    for row in range(5):
      for col in range(5):
        s = row * 5 + col

        if self.s == s:
          print("X", end="")
        elif Environment5x5._is_obsticle(s):
          print("V", end="")
        elif Environment5x5._is_goal(s):
            print("G", end="")
        else:
          print("_", end="")
      print("")
    print("")

  def _is_termial(s):
    return s in [12]

  def _is_obsticle(s):
    return s in [7, 11]

  def _is_goal(s):
    return s in [12]

  def _get_reward(s):
    if Environment5x5._is_obsticle(s):
      return -1
    elif Environment5x5._is_goal(s):
      return 1
    else:
      return 0