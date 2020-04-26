import numpy as np
import random
from envs.e5x5.Environment5x5 import Environment5x5
from envs.e5x5moving.Environment5x5Moving import Environment5x5Moving

def make(enviromnent):
  """Supported environments are
  - 5x5
  - 5x5moving
  """
  if enviromnent == '5x5':
    return Environment5x5()
  elif enviromnent == '5x5moving':
    return Environment5x5Moving()
  
  raise ValueError


  
