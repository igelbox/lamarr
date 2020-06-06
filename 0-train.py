from setup_gym import gym
from modules.training import TrainingGym

gym.load()
gym.env.render(mode='human')

def format_value(value):
  if isinstance(value, float):
    return f'{value:.2f}'
  return value

gym = TrainingGym(gym)
gym.load()

while True:
  gym.train_game()
  print({ k: format_value(v[-1]) for k, v, in gym.metrics.items()})
