from time import time
import numpy as np
from os.path import isfile
import pickle

from . import storage
from .environments import RobotsEnvironment
from .gym import Gym

def scale2d(arr, shape):
  h, w = shape
  oh, ow = len(arr), len(arr[0])
  return [
    [arr[oh * y // h][ow * x // w] for x in range(w)]
    for y in range(h)
  ]

def scale1d(arr, size):
  os = len(arr)
  return [
    arr[os * i // size]
    for i in range(size)
  ]

def geneMask(shape):
  gene_shape = tuple(int(np.ceil(np.sqrt(s))) for s in shape)
  gene = np.where(np.random.rand(*gene_shape) > 0.5, 0, 1)
  mask = scale1d(gene, shape[0]) if len(gene.shape) == 1 else scale2d(gene, shape)
  return np.array(mask)

def mixInto(a, b, target):
  def mix_weights(w0, w1, i):
    k = geneMask(w0.shape)
    w = w0 + (w1 - w0) * k
    w *= np.where(np.random.rand(*w.shape) < 0.01, 0, 1)
    r = np.where(np.random.rand(*w.shape) > 0.05, 0, 1)
    w += (np.random.rand(*w.shape) * 2 - 1) * r * 0.1
    return w

  for i, (la, lb, lt) in enumerate(zip(a.layers, b.layers, target.layers)):
    lt.set_weights([
      mix_weights(*w, i)
      for w in zip(la.get_weights(), lb.get_weights())
    ])

def robots_positions(env: RobotsEnvironment):
  return np.array([robot.robot_body.pose().xyz() for robot in env.robots])

class TrainingGym:
  def __init__(self, gym: Gym):
    self.gym = gym

    self.fname_metrics = f'{gym.fdim}-metrics.pkl'
    self.metrics = {}
    self.best_reward = 0

  def load(self):
    fpath_metrics = storage.path + self.fname_metrics
    if isfile(fpath_metrics):
      print(f'loading metrics: {fpath_metrics} ...')
      with open(fpath_metrics, 'rb') as file:
        self.metrics = pickle.load(file)
      rewards = self.metrics.get('reward', [])
      self.best_reward = np.max(rewards, initial=0)
      print(f'best_reward: {self.best_reward}')

  def train_game(self, steps_count=200):
    start_time = time()

    gym = self.gym
    env, agents = gym.env, gym.agents
    agents_count = len(agents)

    states = gym.reset()

    tremor_penalties = np.zeros(shape=(agents_count,))
    z_bonuses = np.zeros(shape=(agents_count,))
    # yaw_penalties = np.zeros(shape=(agents_count,))
    start_positions = robots_positions(env)
    last_actions = np.zeros(shape=(3, *env.action_space.shape))

    for step in range(steps_count):
      states, actions = gym.step(states)

      positions = robots_positions(env)
      last_actions = np.vstack((last_actions[1:], [actions]))
      for i, robot in enumerate(env.robots):
        la0, la1, la2 = last_actions
        acc0, acc1 = la1[i] - la0[i], la2[i] - la1[i]
        tremor_penalties[i] += np.sum(np.where((acc1 * acc0) < 0, 1, 0)) * 0.001
        tremor_penalties[i] += np.sum(np.abs(acc0 - acc1)) * 0.001
        # parts_xyz = np.array([p.pose().xyz() for p in robot.parts.values()]).flatten()
        # z_bonuses[i] += parts_xyz[2::3].mean() * 0.01
        # rpy = robot.robot_body.pose().rpy()
        # yaw_penalties[i] += np.abs(rpy[2]) * 0.01
      z_bonuses += 0.1 / (np.abs(0.4 - positions[:,2]) + 0.1) * 0.001

    # done
    distances = (positions - start_positions)[:,1]
    robot_rewards = distances - tremor_penalties + z_bonuses
    # rewards -= yaw_penalties
    ars = np.argsort(robot_rewards)
    i_winner = ars[-1]
    breed_probabilities = robot_rewards - np.min(robot_rewards)
    breed_probabilities /= np.sum(breed_probabilities)
    for i in range(len(agents) // 2):
      [a, b] = np.random.choice(agents, size=2, p=breed_probabilities)
      mixInto(a, b, agents[ars[i]])

    game_time = time() - start_time

    max_reward = robot_rewards[i_winner]
    if max_reward > self.best_reward:
      self.best_reward = max_reward
      print(f'saving {max_reward} ...')
      gym.save()

    rewards = self.metrics.get('reward', [])
    games_count = len(rewards) + 1
    mean100 = np.mean((*rewards[-101:], max_reward))
    for [k, v] in dict(
      reward=max_reward,
      mean_100=mean100,
      winner=i_winner,
      distance=distances[i_winner],
      tremor_penalty=tremor_penalties[i_winner],
      z_bonus=z_bonuses[i_winner],
      # yaw_penalty=yaw_penalties[iwin],
      time=game_time,
    ).items():
      metric = self.metrics.get(k)
      if metric is None:
        self.metrics[k] = metric = []
      metric.append(v)

    if not games_count % 10:
      fpath_metrics = storage.path + self.fname_metrics
      with open(fpath_metrics, 'wb') as file:
        pickle.dump(self.metrics, file)
