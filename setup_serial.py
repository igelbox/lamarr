import numpy as np
import serial
import struct

from typing import List

RANGE = 45
OFFSET = 0

REMAPPINGS = [
  ((-15, -50), 3, (90+RANGE+OFFSET, 90-RANGE+OFFSET)),
  ((-15, -50), 3, (90+RANGE+OFFSET, 90-RANGE+OFFSET)),
  ((+20, -20), 2, (90, 0)),

  ((-15, -50), 5, (90-RANGE-OFFSET, 90+RANGE-OFFSET)),
  ((-15, -50), 5, (90-RANGE-OFFSET, 90+RANGE-OFFSET)),
  ((-20, +20), 4, (90, 180)),

  ((+15, +50), 7, (90+RANGE+OFFSET, 90-RANGE+OFFSET)),
  ((+15, +50), 7, (90+RANGE+OFFSET, 90-RANGE+OFFSET)),
  ((+20, -20), 6, (90, 0)),

  ((+15, +50), 1, (90-RANGE-OFFSET, 90+RANGE-OFFSET)),
  ((+15, +50), 1, (90-RANGE-OFFSET, 90+RANGE-OFFSET)),
  ((-20, +20), 0, (90, 180)),
]

def remap(remapping, poses):
  s, i, t = remapping
  r = int((poses[i] - s[0]) / (s[1] - s[0]) * (t[1] - t[0]) + t[0])
  return np.clip(r, min(*t), max(*t))

port = serial.Serial('/dev/rfcomm0', baudrate=57600)

def serial_send(action: List[float]):
  spos = [remap(r, action * 90) for r in REMAPPINGS]
  data = struct.pack('<hhhhhhhhhhhh', *spos)
  data = b'\xff' + data.replace(b'\xff', b'\xfe')
  port.write(data)
  port.flush()
  if port.inWaiting():
    strs = port.readline().decode()
    print('>', strs)
