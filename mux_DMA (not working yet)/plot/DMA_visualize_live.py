import serial, struct, numpy as np, matplotlib.pyplot as plt, matplotlib.animation as animation

PORT = '/dev/cu.usbmodem150585901'
BAUD = 115200
ser = serial.Serial(PORT, BAUD)

while True:
    if ser.read(4) == b'DATA':
        break

FRAMES = struct.unpack('i', ser.read(4))[0]
ROWS   = struct.unpack('i', ser.read(4))[0]
COLS   = struct.unpack('i', ser.read(4))[0]
print(f"Frames={FRAMES}, Rows={ROWS}, Cols={COLS}")

raw = ser.read(FRAMES * ROWS * COLS)
data = np.frombuffer(raw, dtype=np.uint8).reshape((FRAMES, ROWS, COLS))
assert ser.read(4) == b'DONE'

fig, ax = plt.subplots()
im = ax.imshow(data[0], cmap='gray', vmin=0, vmax=1)

#def update(i):
 #   im.set_array(data[i])
  #  ax.set_title(f"Frame {i}")
   # return [im]

ani = animation.FuncAnimation(fig, im, frames=FRAMES, interval=50, blit=False)
plt.show()


