import numpy as np

def stack_times(array, count):
  return np.append([array], [array for _ in range(count - 1)], axis=0)
