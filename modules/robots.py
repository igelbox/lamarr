import numpy as np
from gym.spaces import Box
import pybullet as pb
from pybullet import POSITION_CONTROL
from pybullet_envs.robot_bases import MJCFBasedRobot


from .arrays import stack_times

import os
dir_path = os.path.dirname(os.path.realpath(__file__))

class ServAnt(MJCFBasedRobot):
  def __init__(
      self,
      base_position=(0, 0, 1),
  ):
    super(ServAnt, self).__init__('servant.xml', 'torso', action_dim=12, obs_dim=0)
    self.base_position = base_position

  def calc_state(self):
    return np.zeros(0)

  def reset(self, bullet_client):

    self._p = bullet_client
    if (self.doneLoading == 0):
      self.ordered_joints = []
      self.doneLoading = 1
      fpath = os.path.join(dir_path, self.model_xml)
      self.objects = self._p.loadMJCF(fpath,
                                    flags=pb.URDF_USE_SELF_COLLISION |
                                    pb.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
      self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
          self._p, self.objects)
    self.robot_specific_reset(self._p)

    return self.calc_state()

  def robot_specific_reset(self, bullet_client):
    for j in self.ordered_joints:
      j.reset_current_position(self.np_random.uniform(low=-0.1, high=0.1), 0)
    self.scene.actor_introduce(self)
    bullet_client.resetBasePositionAndOrientation(self.objects[0], self.base_position, [0, 0, 0, 1])

  def step(self, action):
    self.apply_action(action)

  def apply_action(self, action):
    for i, joint in enumerate(self.ordered_joints):
      position = np.clip(action[i], -1, +1) * np.pi
      joint._p.setJointMotorControl2(
        joint.bodies[joint.bodyIndex],
          joint.jointIndex,
          POSITION_CONTROL,
          targetPosition=position,
          positionGain=0.04,
          force=100
        )

def withMemory(BaseRobot):
  class WithMemory(BaseRobot):
    def __init__(self, *args,
      last_states_count=3,
      last_states_gap=5,
      **kvargs,
    ):
      super(WithMemory, self).__init__(*args, **kvargs)

      space = self.observation_space
      self.original_observation_space_shape = space.shape
      assert isinstance(space, Box)
      self.observation_space = Box(
          low=stack_times(space.low, last_states_count),
          high=stack_times(space.high, last_states_count),
          dtype=space.dtype,
      )

      self.last_states_count = last_states_count
      self.last_states_gap = last_states_gap

    def reset(self, *args, **kvargs):
      self.last_states = np.zeros(shape=(
          self.last_states_count * self.last_states_gap,
          *self.original_observation_space_shape,
      ))
      return super(WithMemory, self).reset(*args, **kvargs)

    def calc_state(self, *args, **kvargs):
      state = super(WithMemory, self).calc_state(*args, **kvargs)
      last_states = np.vstack((self.last_states[1:], state))
      self.last_states = last_states
      return last_states[::self.last_states_gap]

  return WithMemory

def withTimer(BaseRobot):
  class WithTimer(BaseRobot):
    def __init__(self, *args,
      timer_periods=[47, 71, 107],
      **kvargs,
    ):
      super(WithTimer, self).__init__(*args, **kvargs)

      space = self.observation_space
      assert isinstance(space, Box)
      self.observation_space = Box(
          low=np.concatenate((space.low.flatten(), np.zeros(len(timer_periods)))),
          high=np.concatenate((space.high.flatten(), np.ones(len(timer_periods)))),
          dtype=space.dtype,
      )

      self.timer_periods = timer_periods

    def reset(self, *args, **kvargs):
      self.step_no = 0
      return super(WithTimer, self).reset(*args, **kvargs)

    def step(self, *args, **kvargs):
      self.step_no += 1
      return super(WithTimer, self).step(*args, **kvargs)

    def calc_state(self, *args, **kvargs):
      state = super(WithTimer, self).calc_state(*args, **kvargs)
      time_features = [(self.step_no % period) / period for period in self.timer_periods]
      return np.concatenate((state.flatten(), time_features))

  return WithTimer

def withLastAction(BaseRobot):
  class WithLastAction(BaseRobot):
    def __init__(self, *args, **kvargs):
      super(WithLastAction, self).__init__(*args, **kvargs)

      observation_space = self.observation_space
      assert isinstance(observation_space, Box)
      action_space = self.action_space
      assert isinstance(action_space, Box)
      self.observation_space = Box(
          low=np.concatenate((observation_space.low.flatten(), action_space.low)),
          high=np.concatenate((observation_space.high.flatten(), action_space.high)),
          dtype=observation_space.dtype,
      )

    def calc_state(self, *args, **kvargs):
      state = super(WithLastAction, self).calc_state(*args, **kvargs)
      return np.concatenate((state.flatten(), self.last_action))

    def reset(self, *args, **kvargs):
      self.last_action = np.zeros(shape=self.action_space.shape)
      return super(WithLastAction, self).reset(*args, **kvargs)

    def apply_action(self, action):
      self.last_action = action
      return super(WithLastAction, self).apply_action(action)

  return WithLastAction
