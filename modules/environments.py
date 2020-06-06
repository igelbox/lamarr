import numpy as np
from gym.spaces import Box
from pybullet import COV_ENABLE_RENDERING
from pybullet_envs.env_bases import MJCFBaseBulletEnv
from pybullet_envs.scene_stadium import SinglePlayerStadiumScene

from .arrays import stack_times

class RobotsEnvironment(MJCFBaseBulletEnv):
  def __init__(self, robots):
    self.robots = robots
    self.stateId = -1
    super(RobotsEnvironment, self).__init__(robots[0])

    observation_space = self.observation_space
    assert isinstance(observation_space, Box)
    self.observation_space = Box(
        low=stack_times(observation_space.low, len(robots)),
        high=stack_times(observation_space.high, len(robots)),
        dtype=observation_space.dtype,
    )
    action_space = self.action_space
    assert isinstance(action_space, Box)
    self.action_space = Box(
        low=stack_times(action_space.low, len(robots)),
        high=stack_times(action_space.high, len(robots)),
        dtype=action_space.dtype,
    )

  def create_single_player_scene(self, bullet_client):
    self.stadium_scene = SinglePlayerStadiumScene(
        bullet_client,
        gravity=9.8,
        timestep=0.0165 / 4,
        frame_skip=4,
    )
    return self.stadium_scene

  def seed(self, seed=None):
    result = super(RobotsEnvironment, self).seed(seed)
    for r in self.robots:
      r.np_random = self.np_random
    return result

  def reset(self):
    if (self.stateId >= 0):
      self._p.restoreState(self.stateId)

    super(RobotsEnvironment, self).reset()
    for robot in self.robots:
      robot.scene = self.scene
    result = np.array([robot.reset(self._p) for robot in self.robots])
    # print('???', result.shape)
    self._p.configureDebugVisualizer(COV_ENABLE_RENDERING, 0)

    for robot in self.robots:
      robot.addToScene(self._p, self.stadium_scene.ground_plane_mjcf)
    self._p.configureDebugVisualizer(COV_ENABLE_RENDERING, 1)

    if (self.stateId < 0):
      self.stateId = self._p.saveState()

    return result

  def step(self, actions):
    for robot, action in zip(self.robots, actions):
      robot.step(action)
    self.scene.global_step()

    return np.array([robot.calc_state() for robot in self.robots]), 0, False, {}
