LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

def action_name(a):
  if a == LEFT:
    return "LEFT"
  elif a == DOWN:
    return "DOWN"
  elif a == RIGHT:
    return "RIGHT"
  elif a == UP:
    return "UP"
  else:
    raise ValueError