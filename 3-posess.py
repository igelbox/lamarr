from time import sleep

from setup_gym import gym
from setup_serial import serial_send

gym.load()
gym.env.render(mode='human')

states = gym.reset()
while True:
  states, actions = gym.step(states)

  serial_send(actions[0])
  sleep(0.03)
