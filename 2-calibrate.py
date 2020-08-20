import numpy as np
from time import sleep
from tkinter import Tk, ttk, Scale, HORIZONTAL

from setup_gym import env
from setup_serial import serial_send


poses = np.zeros(env.action_space.shape[1])
tk = Tk()
tk.title('Calibrate')
for i, pose in enumerate(poses):
  def mk(i):
    def command(pos):
      poses[i] = pos

    return command

  w = Scale(tk, from_=-180, to=180, orient=HORIZONTAL, command=mk(i))
  w.set(pose)
  w.pack(fill='both', expand=True)

env.render(mode='human')
env.reset()
while True:
  actions = [(poses) / 180 for _ in env.robots]
  states, *unused = env.step(actions)

  serial_send(actions[0])

  tk.update_idletasks()
  tk.update()
  sleep(0.02)
