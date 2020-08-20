import numpy as np
import serial
import struct

from typing import List

RANGE = 45
OFFSET = 0

REMAPPINGS = [
  ((-60, 45), 4, (90+60, 90-45)),
  ((135, 0), 5, (90+45+10, 90-90+10)),
  ((0, 90), 3, (90, 0)),

  ((-60, 45), 10, (90-60, 90+45)),
  ((135, 0), 11, (90-45, 90+90)),
  ((0, 90), 9, (90, 180)),

  ((-60, 45), 7, (90+60, 90-45)),
  ((135, 0), 8, (90+45+10, 90-90+5)),
  ((0, 90), 6, (90+10, 0+10)),

  ((-60, 45), 1, (90-60, 90+45)),
  ((135, 0), 2, (90-45+10, 90+90+10)),
  ((0, 90), 0, (90-10, 180-10)),
]

def remap(remapping, poses):
  s, i, t = remapping
  r = int((poses[i] - s[0]) / (s[1] - s[0]) * (t[1] - t[0]) + t[0])
  return np.clip(r, min(*t), max(*t))

port = serial.Serial('/dev/rfcomm0', baudrate=57600)

def serial_send(action: List[float]):
  spos = [remap(r, action * 180) for r in REMAPPINGS]
  data = struct.pack('<hhhhhhhhhhhh', *spos)
  data = b'\xff' + data.replace(b'\xff', b'\xfe')
  port.write(data)
  port.flush()
  if port.inWaiting():
    strs = port.readline().decode()
    print('>', strs)
