from setup_gym import gym

gym.load()
gym.env.render(mode='human')

states = gym.reset()
while True:
  states, *unused = gym.step(states)
