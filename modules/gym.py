import numpy as np
from os.path import isfile

from . import storage

class Gym:
  def __init__(self, env, agents, model, fdim):
    self.env = env
    self.agents = agents
    self.model = model
    self.fdim = fdim
    self.fname_model = f'{fdim}-model.h5'

  def load(self):
    fpath_model = storage.path + self.fname_model
    if isfile(fpath_model):
      print(f'loading model: {fpath_model} ...')
      self.model.load_weights(fpath_model)

  def save(self):
    fpath_model = storage.path + self.fname_model
    self.model.save(fpath_model)

  def reset(self):
    return self.env.reset()

  def step(self, states):
    states = states.reshape(len(states), 1, self.model.input_shape[0][1])
    actions = self.model.predict(list(states))
    actions = np.array(actions).reshape(*self.env.action_space.shape)
    states, *unused = self.env.step(actions)
    return states, actions

